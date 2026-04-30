import pickle, sys
path = sys.argv[1]
data = pickle.load(open(path, "rb"))
print("path:", path)
print("keys:", list(data.keys()))
exprs = data.get("exprs") or data.get("generators")
print("n exprs:", len(exprs))
for i in [0, 1, 2, 100, 155]:
    if i < len(exprs):
        print(f"  expr[{i}]:", repr(exprs[i])[:200])
syms = set()
for e in exprs[:30]:
    syms |= e.free_symbols
print("syms in first 30:", sorted(map(str, syms)))
print("levels dist:", {l: data.get("levels", [0]).count(l) for l in set(data.get("levels", [0]))})
print("names sample:", data.get("names", ["n/a"])[:8])
