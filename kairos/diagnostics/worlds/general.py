"""Random made-up worlds with mixed variable types and many forms of equation. Written before looking at any result; not tuned."""
import numpy as np
def st(a): return (a-a.mean())/max(a.std(), 1e-12)
def curve(rng):
    k = rng.integers(0, 5); s = rng.choice([-1, 1])*rng.uniform(.7, 1.3)
    return [lambda x: s*x, lambda x: s*1.3*np.tanh(1.5*x), lambda x: s*np.sin(1.8*x), lambda x: s*.7*(x**2-1), lambda x: s*np.exp(np.clip(x, -2, 1.6))*.6][k]
def noise(rng, n):
    k = rng.integers(0, 4)
    return [rng.normal(size=n), rng.uniform(-1.7, 1.7, n), rng.gamma(2, 1, n)/1.41-1.41, rng.laplace(size=n)/1.41][k]
ROOTS = ["normal", "uniform", "lognormal", "beta", "bimodal", "binary", "ordinal", "nominal"]
CONT = ["additive", "exp", "logistic", "gate", "product", "formula", "spread", "floor", "count"]
def make(seed, n=1000, size=10):
    rng = np.random.default_rng(1000+seed); data, kinds, scales, truth, how = {}, {}, {}, {}, {}
    z = {}  # standardised continuous values, codes for categorical
    for i in range(size):
        v = f"v{i}"; earlier = [f"v{j}" for j in range(i)]
        np_ = 0 if i < 2 else int(min(len(earlier), rng.choice([0, 1, 2, 3, 4], p=[.2, .15, .35, .2, .1])))
        ps = list(rng.choice(earlier, np_, replace=False)) if np_ else []
        if not ps:
            r = ROOTS[rng.integers(0, 5)] if rng.random() < .65 else ROOTS[rng.integers(5, 8)]
            how[v] = r; kinds[v] = r if r in ("binary", "ordinal", "nominal") else "continuous"
            x = {"normal": lambda: rng.normal(size=n), "uniform": lambda: rng.uniform(0, 1, n), "lognormal": lambda: np.exp(rng.uniform(.4, .9)*rng.normal(size=n)),
                 "beta": lambda: rng.beta(3, 1.5, n), "bimodal": lambda: np.where(rng.random(n) < .6, rng.normal(-1, .4, n), rng.normal(1, .5, n)),
                 "binary": lambda: (rng.random(n) < rng.uniform(.3, .7))*1., "ordinal": lambda: rng.choice(5, n, p=[.1, .25, .3, .25, .1])*1., "nominal": lambda: rng.choice(4, n, p=[.3, .3, .2, .2])*1.}[r]()
        else:
            cat = [p for p in ps if kinds[p] != "continuous"]; con = [p for p in ps if kinds[p] == "continuous"]
            def part(p):
                if kinds[p] == "continuous": return curve(rng)(z[p])
                lv = np.unique(z[p]); h = rng.normal(size=len(lv)) if kinds[p] == "nominal" else np.cumsum(rng.uniform(0, 1, len(lv)))*rng.choice([-1, 1])
                h = (h-h.mean())/max(h.std(), .3); return h[np.searchsorted(lv, z[p])]
            s = sum(part(p) for p in ps); s = st(s); e = noise(rng, n); sd = rng.choice([.1, .25, .5])
            t = rng.choice(["binary", "ordinal", "nominal"], p=[.5, .3, .2]) if rng.random() < .3 else "cont"
            if t == "binary": x = (rng.random(n) < 1/(1+np.exp(-2.2*s)))*1.; how[v] = kinds[v] = "binary"
            elif t == "ordinal":
                u = s + .5*rng.logistic(size=n)*.6; x = np.searchsorted(np.quantile(u, [.2, .45, .7, .9]), u)*1.; how[v] = kinds[v] = "ordinal"
            elif t == "nominal":
                sc = np.column_stack([np.zeros(n)] + [2*s*rng.choice([-1, 1]) + sum(part(p) for p in ps)*.7 for _ in range(2)]); g = sc + rng.gumbel(size=sc.shape); x = g.argmax(1)*1.; how[v] = kinds[v] = "nominal"
            else:
                kinds[v] = "continuous"; form = rng.choice(CONT)
                if cat and con and rng.random() < .35: form = "gate"
                if len(con) >= 2 and form == "additive" and rng.random() < .4: form = "product"
                if form == "gate" and not (cat and con): form = "additive"
                if form in ("product", "formula") and len(con) < 2: form = "exp"
                if form == "count" and sd > .3: form = "additive"
                how[v] = str(form)
                if form == "additive": x = s + sd*e
                elif form == "exp": x = np.exp(.8*s + .8*sd*e)
                elif form == "logistic": x = 1/(1+np.exp(-(2*s - .5 + 2*sd*e)))
                elif form == "gate":
                    on = (z[cat[0]] == np.unique(z[cat[0]])[-1]) | (z[cat[0]] == np.unique(z[cat[0]])[0])*(len(np.unique(z[cat[0]])) > 2)
                    x = on*(1.5 + st(sum(part(p) for p in con)) ) + sd*.5*e
                elif form == "product": x = (z[con[0]]+rng.uniform(-.5, .5))*(z[con[1]]+rng.uniform(-.5, .5)) + .5*sum((part(p) for p in ps if p not in con[:2]), np.zeros(n)) + sd*e
                elif form == "formula":
                    x = data[con[0]]*rng.choice([1, 2])/data[con[0]].std() - data[con[1]]/data[con[1]].std()*rng.uniform(.5, 1.5) + .7*sum((part(p) for p in ps if p not in con[:2]), np.zeros(n)); x = np.round(x, 2)
                elif form == "spread": x = s + sd*e*np.exp(.6*st(part(ps[0])))
                elif form == "floor": x = np.maximum(s + 1, .2)*np.exp(.5*sd*rng.normal(size=n))
                elif form == "count": x = rng.poisson(np.exp(1.5 + .8*s))*1.
        data[v] = x; truth[v] = ps; z[v] = st(x) if kinds[v] == "continuous" else x
        scales[v] = {"exp": "positive", "logistic": "probability", "floor": "positive", "count": "count", "lognormal": "positive", "beta": "probability"}.get(how[v], "any")
    # shuffled names so order carries no hint
    perm = rng.permutation(size); name = {f"v{i}": f"x{perm[i]}" for i in range(size)}
    f = lambda d: {name[k]: d[k] for k in d}
    return f(data), f(kinds), f(scales), {name[k]: sorted(name[p] for p in ps) for k, ps in truth.items()}, f(how)
if __name__ == "__main__":
    from collections import Counter
    c = Counter(); e = 0
    for s in range(24):
        d, k, sc, t, h = make(s); c.update(h.values()); e += sum(map(len, t.values()))
        assert all(np.isfinite(x).all() for x in d.values()), s
    print(c, e/24)
