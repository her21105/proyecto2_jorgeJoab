#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CYK.py — Reconocimiento y árbol(es) de parseo con el algoritmo CYK sobre una CNF.

Lee una gramática en CNF (como la que genera FNC.py en cnf.json),
recibe una oración (tokens separados por espacios) y decide pertenencia.
Además, reconstruye al menos un árbol sintáctico (y opcionalmente todos hasta un límite).

CLI:
  python CYK.py --cnf cnf.json --sent "she eats a cake" [--all] [--max_trees 20] [--dot tree.dot]
  python CYK.py --cnf cnf.json --file sentences.txt [--all] [--max_trees 20]

Salida:
  - Imprime SI/NO, tiempo de ejecución y, si pertenece, un árbol en forma parentizada.
  - Con --all imprime hasta --max_trees árboles.
  - Con --dot exporta el primer árbol en formato Graphviz DOT.

Notas:
  - La CNF debe contener reglas A->a o A->BC exclusivamente (salvo ε en start si aplica).
  - Tokenización sencilla: separa por espacios; convierte tokens a minúsculas por defecto
    usando --lower (activo por defecto) / --no-lower para desactivar.
"""

from __future__ import annotations
import argparse
import json
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Any
import subprocess
from graphviz import Digraph
import os


def tree_to_graphviz_obj(t) -> Digraph:
    g = Digraph('G', node_attr={'shape': 'plain'})
    counter = {'i': 0}

    def new_id():
        i = counter['i']
        counter['i'] += 1
        return f"n{i}"

    def walk(node):
        # Hoja: (A, palabra)
        if len(node) == 2 and isinstance(node[1], str):
            A, w = node
            nA = new_id()
            nw = new_id()
            g.node(nA, A)
            g.node(nw, w)
            g.edge(nA, nw)
            return nA
        # Interno: (A, L, R)
        A, L, R = node
        nA = new_id()
        g.node(nA, A)
        nL = walk(L)
        nR = walk(R)
        g.edge(nA, nL)
        g.edge(nA, nR)
        return nA

    if t:
        walk(t)
    return g

# --------------------------- Lectura CNF ---------------------------
class CNF:
    def __init__(self, start: str, terminal_rules: Dict[str, Set[str]], binary_rules: Dict[Tuple[str,str], Set[str]],
                 nonterminals: Set[str], terminals: Set[str]):
        self.start = start
        self.terminal_rules = terminal_rules  # a -> {A}
        self.binary_rules = binary_rules      # (B,C) -> {A}
        self.nonterminals = nonterminals
        self.terminals = terminals

    @staticmethod
    def from_json(path: str) -> 'CNF':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        rules = data['rules']
        start = data['start']
        nonterminals = set(data.get('nonterminals', rules.keys()))
        terminals = set(data.get('terminals', []))

        terminal_rules: Dict[str, Set[str]] = defaultdict(set)
        binary_rules: Dict[Tuple[str,str], Set[str]] = defaultdict(set)

        for A, rhss in rules.items():
            for rhs in rhss:
                if len(rhs) == 1:
                    a = rhs[0]
                    # A -> a
                    terminal_rules[a].add(A)
                elif len(rhs) == 2:
                    B, C = rhs
                    binary_rules[(B, C)].add(A)
                elif len(rhs) == 0:
                    # epsilon (permitido solo para start, pero CYK estándar no lo maneja)
                    pass
                else:
                    raise ValueError(f"Regla no CNF: {A} -> {rhs}")
        return CNF(start, terminal_rules, binary_rules, nonterminals, terminals)

# --------------------------- CYK + Backpointers ---------------------------
class Backptr:
    """Representa cómo se obtuvo A en T[i,j].
       - Si terminal: kind='T', token=word
       - Si binaria:  kind='B', split=k, left=B, right=C
    """
    __slots__ = ('kind','token','split','left','right')
    def __init__(self, kind: str, token: Optional[str]=None, split: Optional[int]=None,
                 left: Optional[str]=None, right: Optional[str]=None):
        self.kind = kind
        self.token = token
        self.split = split
        self.left = left
        self.right = right

class CYKParser:
    def __init__(self, cnf: CNF):
        self.cnf = cnf

    def parse(self, words: List[str]):
        n = len(words)
        if n == 0:
            # caso vacío: fuera de alcance de CYK clásico, salvo si start -> ε
            return [], []
        # T[i][l]: dict NT -> list[Backptr]; i = inicio, l = longitud
        T: List[List[Dict[str, List[Backptr]]]] = [ [defaultdict(list) for _ in range(n+1)] for _ in range(n) ]

        # Base: longitud 1
        for i, w in enumerate(words):
            for A in self.cnf.terminal_rules.get(w, set()):
                T[i][1][A].append(Backptr('T', token=w))

        # Longitudes 2..n
        for l in range(2, n+1):         # l = longitud de la subcadena
            for i in range(0, n-l+1):   # i = inicio
                for k in range(1, l):   # split en k (left length = k)
                    left_cell = T[i][k]
                    right_cell = T[i+k][l-k]
                    if not left_cell or not right_cell:
                        continue
                    for B in left_cell.keys():
                        for C in right_cell.keys():
                            for A in self.cnf.binary_rules.get((B, C), set()):
                                T[i][l][A].append(Backptr('B', split=k, left=B, right=C))
        return T, words

    # Reconstrucción de árboles ----------------------------------------------------
    def build_all_trees(self, T, words, max_trees: int = 20) -> List[Any]:
        n = len(words)
        start = self.cnf.start
        if n == 0:
            return []
        cell = T[0][n]
        if start not in cell:
            return []
        trees: List[Any] = []
        def dfs(i: int, l: int, A: str, quota: List[int]):
            if quota[0] <= 0:
                return
            for bp in T[i][l][A]:
                if bp.kind == 'T':
                    trees.append((A, words[i]))
                    quota[0] -= 1
                elif bp.kind == 'B':
                    k = bp.split
                    for left_tree in self._expand(i, k, bp.left, T, words, quota):
                        if quota[0] <= 0:
                            return
                        for right_tree in self._expand(i+k, l-k, bp.right, T, words, quota):
                            if quota[0] <= 0:
                                return
                            trees.append((A, left_tree, right_tree))
                            quota[0] -= 1
        # La expansión devuelve generadores de subárboles
        # Para controlar el crecimiento combinatorio, generamos de manera lazy
        # y utilizamos un límite global de árboles.
        return self._collect_trees(T, words, start, max_trees)

    def _collect_trees(self, T, words, A0: str, max_trees: int) -> List[Any]:
        n = len(words)
        res: List[Any] = []
        def gen(i, l, A):
            for bp in T[i][l][A]:
                if bp.kind == 'T':
                    yield (A, words[i])
                else:
                    k = bp.split
                    for L in gen(i, k, bp.left):
                        for R in gen(i+k, l-k, bp.right):
                            yield (A, L, R)
        if A0 not in T[0][n]:
            return []
        g = gen(0, n, A0)
        try:
            for _ in range(max_trees):
                res.append(next(g))
        except StopIteration:
            pass
        return res

    # Pretty-printers -------------------------------------------------------------
    def tree_to_parenthesized(self, t) -> str:
        if isinstance(t, tuple):
            if len(t) == 2 and isinstance(t[1], str):
                A, w = t
                return f"({A} {w})"
            elif len(t) == 3:
                A, L, R = t
                return f"({A} {self.tree_to_parenthesized(L)} {self.tree_to_parenthesized(R)})"
        return str(t)

    def tree_to_dot(self, t) -> str:
        lines = ["digraph G {", "node [shape=plain];"]
        counter = {'i': 0}
        def node(label):
            i = counter['i']; counter['i'] += 1
            name = f"n{i}"
            safe = label.replace('"','\\"')
            lines.append(f'{name} [label="{safe}"];')
            return name
        def walk(t):
            if len(t) == 2 and isinstance(t[1], str):
                A, w = t
                nA = node(A)
                nw = node(w)
                lines.append(f"{nA} -> {nw};")
                return nA
            else:
                A, L, R = t
                nA = node(A)
                nL = walk(L)
                nR = walk(R)
                lines.append(f"{nA} -> {nL};")
                lines.append(f"{nA} -> {nR};")
                return nA
        if t:
            walk(t)
        lines.append("}")
        return "\n".join(lines)

# --------------------------- CLI ---------------------------

def run_once(cnf_path: str, sent: str, all_trees: bool, max_trees: int, lower: bool, dot_path: Optional[str]):
    cnf = CNF.from_json(cnf_path)
    words = sent.strip().split()
    if lower:
        words = [w.lower() for w in words]

    parser = CYKParser(cnf)
    t0 = time.perf_counter()
    T, toks = parser.parse(words)
    dt = time.perf_counter() - t0

    n = len(toks)
    belongs = (n > 0 and cnf.start in T[0][n])

    print(f"Frase: {' '.join(sent.split())}")
    print(f"Resultado: {'SI' if belongs else 'NO'}")
    print(f"Tiempo: {dt*1000:.3f} ms\n")

    if belongs:
        trees = parser._collect_trees(T, toks, cnf.start, max_trees if all_trees else 1)
        if not trees:
            print("(No se pudieron reconstruir árboles; ¿CNF con ε-start?)")
            return
        if all_trees:
            print(f"Árboles ({len(trees)} mostrados, límite {max_trees}):")
            for i, tr in enumerate(trees, 1):
                print(f"[{i}] {parser.tree_to_parenthesized(tr)}")
        else:
            print("Árbol:")
            print(parser.tree_to_parenthesized(trees[0]))

        if dot_path:
            for i, tr in enumerate(trees, 0):
                out_dir = dot_path
                basename = 'arbol'
                base = os.path.join(out_dir, basename + (f"_{i+1}" if all_trees else ""))

                # Construir el grafo desde el árbol
                g = tree_to_graphviz_obj(tr)

                # Exportar PNG (y opcionalmente SVG)
                png_path = g.render(filename=base, format='png', cleanup=True)  # crea base.png
                # svg_path = g.render(filename=base, format='svg', cleanup=False)  # si quieres SVG también

                print(f"Imagen exportada a: {png_path}")


def main():

    cnf = r'C:\Users\Joabh\Documents\GitHub\proyecto2_jorgeJoab\cnf.json' #ruta al cnf.json generado por FNC.py
    sent = "the cat eats the cake with a fork in the oven" #oración a verificar (tokens separados por espacios)
    file = None #archivo con una oración por línea
    all_trees = True #imprimir todos los árboles hasta --max_trees
    max_trees = 20 #límite de árboles a imprimir cuando --all
    lower = True #convertir tokens a minúsculas
    dot = r'C:\Users\Joabh\Documents\GitHub\proyecto2_jorgeJoab\arboles' #ruta de (carpeta) salida para exportar el primer árbol en formato DOT



    #ap = argparse.ArgumentParser(description="Algoritmo CYK con reconstrucción de árboles (CNF)")
    #ap.add_argument('--cnf', required=True, help='Ruta al cnf.json generado por FNC.py')
    #group = ap.add_mutually_exclusive_group(required=True)
    #group.add_argument('--sent', help='Oración a verificar (tokens separados por espacios)')
    #group.add_argument('--file', help='Archivo con una oración por línea')
    #ap.add_argument('--all', action='store_true', help='Imprimir todos los árboles hasta --max_trees')
    #ap.add_argument('--max_trees', type=int, default=20, help='Límite de árboles a imprimir cuando --all')
    #ap.add_argument('--no-lower', dest='lower', action='store_false', help='No convertir tokens a minúsculas')
    #ap.add_argument('--dot', help='Ruta de salida para exportar el primer árbol en formato DOT')
    #args = ap.parse_args()


    if sent:
        run_once(cnf, sent, all_trees, max_trees, lower, dot)
    else:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                run_once(cnf, line, all_trees, max_trees, lower, dot)
                print('-'*60)

if __name__ == '__main__':
    main()
