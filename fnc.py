#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FNC.py — Conversión de una CFG a Forma Normal de Chomsky (FNC/CNF).

Entrada: archivo de gramática en texto plano con formato:
  - Separador de producción: "\\arrow" (con o sin espacios alrededor)
  - Alternativas separadas por "|"
  - Tokens en el lado derecho separados por espacios
  - Comentarios con "#" y líneas en blanco permitidas

Ejemplos:
  S\arrowNP VP
  VP \arrow V NP | VP PP | eats | cuts
  NP \arrow Det N | she | cake

CLI:
  python FNC.py --in grammar.txt --out cnf.json --txt cnf.txt --report report.md [--start S]

Salidas:
  - cnf.json : gramática en CNF estructurada (JSON)
  - cnf.txt  : gramática en CNF legible (mismo estilo con "\\arrow" y "|")
  - report.md: resumen de pasos y estadísticas

Notas:
  - Se agrega un símbolo inicial fresco S0 si el start especificado no es único o
    si queremos preservar que S no aparezca en ningún RHS (estándar para CNF).
  - Se eliminan producciones ε (excepto S0 -> ε si ∈ L(G)), unitarias, símbolos inútiles.
  - Se aplican "terminal lifting" y binarización para cumplir A->BC o A->a.
  - Se emiten "mappings" para rastrear transformaciones (útiles para reconstruir parse trees en CYK).
"""

from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple

ARROW = r"\\arrow"  # patrón literal
NT_PATTERN = re.compile(r"^[A-Z][A-Za-z0-9_]*$")  # Heurística: NT en Mayúscula inicial

# ------------------------------ Utilidades de gramática ------------------------------
class Grammar:
    def __init__(self, start: str | None = None):
        self.nonterminals: Set[str] = set()
        self.terminals: Set[str] = set()
        self.rules: Dict[str, List[List[str]]] = defaultdict(list)  # A -> [[...], ...]
        self.start: str | None = start

    def add_rule(self, lhs: str, rhs: List[str]):
        self.nonterminals.add(lhs)
        # Clasificar símbolos RHS
        for sym in rhs:
            if NT_PATTERN.match(sym):
                self.nonterminals.add(sym)
            else:
                self.terminals.add(sym)
        self.rules[lhs].append(rhs)

    @staticmethod
    def parse_txt(text: str, start_hint: str | None = None) -> "Grammar":
        """Parsea el formato personalizado con \arrow y |. Acepta espacios opcionales."""
        g = Grammar()
        first_lhs = None
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            # dividir por \arrow (con o sin espacios)
            parts = re.split(ARROW, line)
            if len(parts) != 2:
                # tolerar posible espacio alrededor
                if ' \\arrow ' in line:
                    lhs, rhs_str = line.split(' \\arrow ', 1)
                elif '\\arrow ' in line:
                    lhs, rhs_str = line.split('\\arrow ', 1)
                elif ' \\arrow' in line:
                    lhs, rhs_str = line.split(' \\arrow', 1)
                else:
                    raise ValueError(f"Línea inválida (no se encontró \\arrow): {raw}")
            else:
                lhs, rhs_str = parts[0].strip(), parts[1].strip()

            if first_lhs is None:
                first_lhs = lhs

            # alternativas separadas por |
            alts = [alt.strip() for alt in rhs_str.split('|')]
            for alt in alts:
                if alt == "":
                    # Interpretar como epsilon explícito
                    rhs = []
                else:
                    rhs = [tok for tok in alt.split() if tok]
                g.add_rule(lhs, rhs)
        # start symbol
        g.start = start_hint or first_lhs
        return g

    def clone(self) -> "Grammar":
        ng = Grammar(self.start)
        ng.nonterminals = set(self.nonterminals)
        ng.terminals = set(self.terminals)
        ng.rules = defaultdict(list, {A: [list(rhs) for rhs in rhss] for A, rhss in self.rules.items()})
        return ng

    def to_json(self, mappings: dict | None = None) -> dict:
        return {
            "nonterminals": sorted(self.nonterminals),
            "terminals": sorted(self.terminals),
            "start": self.start,
            "rules": {A: rhss for A, rhss in self.rules.items()},
            "mappings": mappings or {},
        }

    def to_txt(self) -> str:
        lines = []
        for A in sorted(self.rules.keys()):
            rhss = self.rules[A]
            parts = []
            for rhs in rhss:
                if len(rhs) == 0:
                    parts.append("")  # epsilon (se imprimirá vacío tras \arrow)
                else:
                    parts.append(" ".join(rhs))
            line = f"{A}\\arrow" + " | ".join(parts)
            lines.append(line)
        return "\n".join(lines)

# ------------------------------ Transformaciones CNF ------------------------------
class CNFConverter:
    def __init__(self, g: Grammar):
        self.g = g.clone()
        self.report = []
        # Mapeos para reconstrucción
        self.mappings = {
            "added_start": None,
            "removed_epsilon": {"nullable": [], "expanded": 0},
            "removed_unit": {"unit_pairs": []},
            "removed_useless": {"removed_non_generating": [], "removed_unreachable": []},
            "terminal_lifting": {},  # terminal -> NewNT
            "binarization": {},      # NewNT -> [X, Y]
        }
        self._fresh_nt_counter = 0
        self._term_nt_cache: Dict[str, str] = {}

    def _fresh_nt(self, base: str = "X") -> str:
        while True:
            self._fresh_nt_counter += 1
            cand = f"{base}{self._fresh_nt_counter}"
            if cand not in self.g.nonterminals and cand not in self.g.terminals:
                self.g.nonterminals.add(cand)
                return cand

    def add_start_symbol(self):
        # Si el start aparece en algún RHS o puede derivar epsilon, agregamos S0 -> S
        S = self.g.start
        assert S is not None
        appears_in_rhs = any(
            S in rhs for rhss in self.g.rules.values() for rhs in rhss
        )
        if appears_in_rhs or True:  # estándar: siempre introducir S0 para seguridad
            S0 = "S0"
            while S0 in self.g.nonterminals or S0 in self.g.terminals:
                S0 = self._fresh_nt("S0")
            self.g.nonterminals.add(S0)
            self.g.rules[S0] = [[S]]
            self.g.start = S0
            self.mappings["added_start"] = {"new_start": S0, "old_start": S}
            self.report.append(f"Símbolo inicial introducido: {S0} → {S}")
        else:
            self.report.append("Símbolo inicial extra no requerido (S no aparece en RHS).")

    # ---- ε-eliminación ----
    def remove_epsilon(self):
        # Encontrar anulables (nullable)
        nullable: Set[str] = set()
        changed = True
        while changed:
            changed = False
            for A, rhss in self.g.rules.items():
                for rhs in rhss:
                    if len(rhs) == 0 or all(sym in nullable for sym in rhs):
                        if A not in nullable:
                            nullable.add(A)
                            changed = True
        # Expandir reglas eliminando símbolos anulables
        new_rules = defaultdict(list)
        expansions = 0
        for A, rhss in self.g.rules.items():
            for rhs in rhss:
                if len(rhs) == 0:
                    # omitimos ε por ahora; lo reintroducimos sólo si el start antiguo lo permite
                    continue
                # construir subconjuntos quitando símbolos anulables (no quitar todos a vacío salvo si A es el start y vacío permitido)
                # Generar todas combinaciones de mantener/quitar símbolos anulables
                positions = [i for i, s in enumerate(rhs) if s in nullable]
                seen: Set[Tuple[str, ...]] = set()
                choices = 1 << len(positions)
                for mask in range(choices):
                    new_rhs = list(rhs)
                    # quitar según máscara
                    for bit, pos in enumerate(positions):
                        if (mask >> bit) & 1:
                            new_rhs[pos] = None
                    new_rhs = tuple([s for s in new_rhs if s is not None])
                    if len(new_rhs) == 0:
                        # sólo permitido si A es símbolo inicial
                        if A == self.g.start:
                            if new_rhs not in seen:
                                new_rules[A].append(list(new_rhs))
                                seen.add(new_rhs)
                                expansions += 1
                        continue
                    if new_rhs not in seen:
                        new_rules[A].append(list(new_rhs))
                        seen.add(new_rhs)
                        expansions += 1
        # Reemplazar
        self.g.rules = new_rules
        self.mappings["removed_epsilon"]["nullable"] = sorted(nullable)
        self.mappings["removed_epsilon"]["expanded"] = expansions
        self.report.append(f"ε-producciones eliminadas. Anulables: {sorted(nullable)}. Nuevas expansiones: {expansions}.")

    # ---- Eliminación de unitarias ----
    def remove_unit(self):
        # Hallar pares unitarios A =>* B con reglas unitarias
        unit_pairs: Set[Tuple[str, str]] = set()
        for A in self.g.nonterminals:
            unit_pairs.add((A, A))
        changed = True
        while changed:
            changed = False
            for A, rhss in self.g.rules.items():
                for rhs in rhss:
                    if len(rhs) == 1 and NT_PATTERN.match(rhs[0]):
                        B = rhs[0]
                        for (C, D) in list(unit_pairs):
                            pass
                        if (A, B) not in unit_pairs:
                            unit_pairs.add((A, B))
                            changed = True

        # Construir nuevas reglas sin unitarias
        new_rules = defaultdict(list)
        for A in self.g.nonterminals:
            # Agregar todas las reglas de B cuyo RHS no sea unitario
            Bs = [B for (X, B) in unit_pairs if X == A]
            seen: Set[Tuple[str, Tuple[str, ...]]] = set()
            for B in Bs:
                for rhs in self.g.rules.get(B, []):
                    if len(rhs) == 1 and NT_PATTERN.match(rhs[0]):
                        continue  # omitir unitarias
                    key = (A, tuple(rhs))
                    if key not in seen:
                        new_rules[A].append(list(rhs))
                        seen.add(key)
        removed_pairs = sorted(list({(A, B) for (A, B) in unit_pairs if A != B}))
        self.g.rules = new_rules
        self.mappings["removed_unit"]["unit_pairs"] = removed_pairs
        self.report.append(f"Producciones unitarias eliminadas. Pares: {removed_pairs}.")

    # ---- Símbolos inútiles ----
    def remove_useless(self):
        # 1) Generativos: pueden derivar una cadena de terminales
        generating: Set[str] = set()
        changed = True
        while changed:
            changed = False
            for A, rhss in self.g.rules.items():
                if A in generating:
                    continue
                for rhs in rhss:
                    if all((not NT_PATTERN.match(s)) or (s in generating) for s in rhs):
                        generating.add(A)
                        changed = True
                        break
        removed_non_generating = sorted(list(self.g.nonterminals - generating))
        # Filtrar reglas por generativos
        self.g.nonterminals &= generating
        self.g.rules = defaultdict(list, {
            A: [rhs for rhs in rhss if all((not NT_PATTERN.match(s)) or (s in generating) for s in rhs)]
            for A, rhss in self.g.rules.items() if A in generating
        })

        # 2) Alcanzables desde el start
        start = self.g.start
        assert start is not None
        reachable_nt: Set[str] = set()
        reachable_t: Set[str] = set()
        queue = deque([start])
        while queue:
            A = queue.popleft()
            if A in reachable_nt:
                continue
            reachable_nt.add(A)
            for rhs in self.g.rules.get(A, []):
                for s in rhs:
                    if NT_PATTERN.match(s):
                        if s not in reachable_nt:
                            queue.append(s)
                    else:
                        reachable_t.add(s)
        removed_unreachable = sorted(list(self.g.nonterminals - reachable_nt))
        self.g.nonterminals &= reachable_nt
        self.g.terminals &= reachable_t
        self.g.rules = defaultdict(list, {A: rhss for A, rhss in self.g.rules.items() if A in reachable_nt})

        self.mappings["removed_useless"]["removed_non_generating"] = removed_non_generating
        self.mappings["removed_useless"]["removed_unreachable"] = removed_unreachable
        self.report.append(
            f"Símbolos inútiles eliminados. No generativos: {removed_non_generating}. No alcanzables: {removed_unreachable}."
        )

    # ---- Terminal lifting: reemplaza terminales en RHS de longitud >= 2 ----
    def terminal_lifting(self):
        new_rules = defaultdict(list)
        for A, rhss in self.g.rules.items():
            for rhs in rhss:
                if len(rhs) >= 2:
                    new_rhs = []
                    for s in rhs:
                        if NT_PATTERN.match(s):
                            new_rhs.append(s)
                        else:
                            # terminal — levantar a NT único por terminal
                            if s not in self._term_nt_cache:
                                T = f"T_{s}"
                                if T in self.g.nonterminals or T in self.g.terminals:
                                    T = self._fresh_nt("T")
                                self._term_nt_cache[s] = T
                                self.g.nonterminals.add(T)
                                # T -> s
                                new_rules[T].append([s])
                                self.mappings["terminal_lifting"][s] = T
                            new_rhs.append(self._term_nt_cache[s])
                    new_rules[A].append(new_rhs)
                else:
                    new_rules[A].append(rhs)
        self.g.rules = new_rules
        self.report.append(
            f"Terminal lifting aplicado. Introducidos {len(self._term_nt_cache)} preterminales."
        )

    # ---- Binarización: descompone RHS de longitud > 2 a binario ----
    def binarize(self):
        new_rules = defaultdict(list)
        for A, rhss in self.g.rules.items():
            for rhs in rhss:
                k = len(rhs)
                if k <= 2:
                    new_rules[A].append(rhs)
                    continue
                # Crear cadenas de NTs intermedios A -> X1 B, X1 -> X2 C, ...
                current_left = A
                symbols = rhs
                # estrategia: asociatividad izquierda
                # A -> s1 X1; X1 -> s2 X2; ...; X_{k-2} -> s_{k-1} s_k
                prev = symbols[0]
                # Creamos una secuencia de nuevos NTs
                for i in range(1, k - 1):
                    Xi = self._fresh_nt("X")
                    new_rules[current_left].append([prev, Xi])
                    self.mappings["binarization"][Xi] = [symbols[i], None]  # rellenaremos después
                    current_left = Xi
                    prev = symbols[i]
                # Última regla
                new_rules[current_left].append([prev, symbols[-1]])
                # Actualizar mapeos de binarización con pares reales
                # Recorremos de nuevo para apuntar pares correctos
                # Nota: los mapeos se usan solo como guía; el árbol final lo reconstruirá CYK siguiendo estas reglas.
        # Completar pares en mapeos
        # (Opcional: ya que arriba dejamos marcadores, aquí reconstruimos desde new_rules)
        self.g.rules = new_rules
        # Rellenar pares reales
        for A, rhss in self.g.rules.items():
            for rhs in rhss:
                if len(rhs) == 2:
                    # rhs = [X, Y] ; si A es fresco X*, registrar
                    if A.startswith('X') or A.startswith('S0X'):
                        self.mappings["binarization"][A] = rhs
        self.report.append("Binarización aplicada (todas las RHS con longitud >2 divididas en reglas binarias).")

    def ensure_cnf_forms(self):
        # Validar que todas las reglas sean A->BC o A->a, y que no existan vacías salvo start si aplica.
        for A, rhss in self.g.rules.items():
            for rhs in rhss:
                if len(rhs) == 1:
                    s = rhs[0]
                    assert not NT_PATTERN.match(s), f"CNF inválida: {A} -> {s} (unitaria)"
                elif len(rhs) == 2:
                    assert all(NT_PATTERN.match(x) for x in rhs), f"CNF inválida: {A} -> {' '.join(rhs)} (no binaria de NTs)"
                elif len(rhs) == 0:
                    assert A == self.g.start, f"Solo el símbolo inicial puede producir ε en CNF: {A}"
                else:
                    raise AssertionError(f"CNF inválida: {A} -> {' '.join(rhs)}")

    def convert(self) -> Tuple[Grammar, dict, str]:
        self.report.append("== Conversión a FNC/CNF iniciada ==")
        self.add_start_symbol()
        self.remove_epsilon()
        self.remove_unit()
        self.remove_useless()
        self.terminal_lifting()
        self.binarize()
        self.remove_useless()  # limpieza final por si quedaron símbolos colgantes
        self.ensure_cnf_forms()
        self.report.append("== Conversión finalizada ==")
        return self.g, self.mappings, "\n".join(self.report)

# ------------------------------ CLI ------------------------------

def main():

    outfile = "cnf.json" #salida JSON (cnf.json)
    txtfile = "cnf.txt" #salida legible (cnf.txt)
    repfile = "report.md" #reporte de pasos (report.md)
    infile = r'C:\Users\Joabh\Documents\GitHub\proyecto2_jorgeJoab\gramatica.txt' #ruta del archivo de gramática (txt)
    start = None #símbolo inicial (por defecto, primer LHS)


    #ap = argparse.ArgumentParser(description="Convertir CFG a FNC/CNF.")
    #ap.add_argument("--in", dest="infile", required=True, help="Ruta del archivo de gramática (txt)")
    #ap.add_argument("--out", dest="outfile", default="cnf.json", help="Salida JSON (cnf.json)")
    #ap.add_argument("--txt", dest="txtfile", default="cnf.txt", help="Salida legible (cnf.txt)")
    #ap.add_argument("--report", dest="repfile", default="report.md", help="Reporte de pasos (report.md)")
    #ap.add_argument("--start", dest="start", default=None, help="Símbolo inicial (por defecto, primer LHS)")
    #args = ap.parse_args()

    with open(infile, "r", encoding="utf-8") as f:
        text = f.read()
    g = Grammar.parse_txt(text, start_hint=start)

    conv = CNFConverter(g)
    cnf_g, mappings, report = conv.convert()

    # Escribir salidas
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(cnf_g.to_json(mappings=mappings), f, ensure_ascii=False, indent=2)
    with open(txtfile, "w", encoding="utf-8") as f:
        f.write(cnf_g.to_txt() + "\n")
    with open(repfile, "w", encoding="utf-8") as f:
        f.write("# CNF Conversion Report\n\n")
        f.write(report + "\n\n")
        f.write("## Resumen\n")
        f.write(f"- No terminales: {len(cnf_g.nonterminals)}\n")
        f.write(f"- Terminales: {len(cnf_g.terminals)}\n")
        f.write(f"- Reglas: {sum(len(rhss) for rhss in cnf_g.rules.values())}\n")
    print("Conversión a FNC/CNF completada.")

if __name__ == "__main__":
    main()
