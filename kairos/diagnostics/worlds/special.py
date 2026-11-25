"""Made-up worlds that stress one thing: exact formulas, nearly straight arrows with non-bell noise, lopsided noise."""
import numpy as np


def formulas(seed, n=1000):
    rng = np.random.default_rng(5000+seed); d, t = {}, {}
    def root(): return [lambda: rng.normal(size=n)*rng.uniform(.3, 3), lambda: rng.uniform(0, rng.uniform(.5, 4), n)+.5, lambda: np.exp(rng.uniform(.3, .7)*rng.normal(size=n)), lambda: rng.gamma(3, 1, n)][rng.integers(0, 4)]()
    for i in range(4): d[f"r{i}"] = root(); t[f"r{i}"] = []
    for i in range(3):
        a, b = rng.choice(sorted(d), 2, replace=False); k = rng.integers(0, 5); A, B = d[a], d[b]
        x = [lambda: A - B, lambda: 2*A + rng.uniform(.3, 3)*B, lambda: A*B, lambda: A/(np.abs(B)+.5), lambda: np.log1p(np.abs(A))*3 + B][k]()
        dec = rng.choice([-1, 2, 4]); x = x if dec < 0 else np.round(x, dec); d[f"f{i}"] = x; t[f"f{i}"] = sorted([a, b])
    for i in range(2):
        a, b = rng.choice(sorted(d), 2, replace=False); A, B = [(d[q]-d[q].mean())/d[q].std() for q in (a, b)]
        d[f"c{i}"] = np.tanh(1.5*A) + .6*B**2*rng.choice([0, 1]) + .6*B + .3*rng.normal(size=n); t[f"c{i}"] = sorted([a, b])
    names = sorted(d); perm = rng.permutation(len(names)); m = {k: f"x{p}" for k, p in zip(names, perm)}
    return {m[k]: x for k, x in d.items()}, {m[k]: sorted(m[p] for p in ps) for k, ps in t.items()}


def mild(seed, count=800, size=8):
    """Nearly straight arrows with strong noise, evenly spread or heavy-tailed."""
    random = np.random.default_rng(seed)
    data, truth = {}, {}
    for child in range(size):
        parents = [p for p in range(child) if random.uniform() < 0.35]
        noise = random.uniform(-1, 1, count) if random.uniform() < 0.5 else random.laplace(0, 0.5, count)
        value = noise + sum(np.tanh(random.uniform(0.5, 1.5) * data[f"v{p}"]) * random.choice([-1, 1]) for p in parents)
        data[f"v{child}"] = (value - value.mean()) / value.std()
        truth[f"v{child}"] = [f"v{p}" for p in parents]
    return data, truth


def skewed(seed, count=800, size=8):
    """The same with lopsided noise (exponential, gamma or lognormal, either way round)."""
    random = np.random.default_rng(7000 + seed)
    data, truth = {}, {}
    for child in range(size):
        parents = [p for p in range(child) if random.uniform() < 0.35]
        kind = random.integers(0, 3)
        noise = [random.exponential(.6, count), random.gamma(2, .4, count), random.lognormal(0, .5, count)][kind] * random.choice([-1, 1])
        value = noise + sum(np.tanh(random.uniform(0.5, 1.5) * data[f"v{p}"]) * random.choice([-1, 1]) for p in parents)
        data[f"v{child}"] = (value - value.mean()) / value.std()
        truth[f"v{child}"] = [f"v{p}" for p in parents]
    return data, truth


def sizes(seed, n=1000, size=8):
    """Noise whose size depends on the prediction: shares, counts, multiplied noise."""
    r = np.random.default_rng(9000+seed); data, truth, z = {}, {}, {}
    for i in range(size):
        v = f"v{i}"; ps = [f"v{j}" for j in range(i) if r.random() < .35]
        if len(ps) > 3: ps = list(r.choice(ps, 3, replace=False))
        if not ps:
            x = [lambda: r.normal(size=n), lambda: r.beta(3, 2, n), lambda: np.exp(.5*r.normal(size=n)), lambda: r.uniform(0, 1, n)][r.integers(0, 4)]()
        else:
            s = sum(r.choice([-1, 1])*r.uniform(.6, 1.2)*[lambda x: x, np.tanh, lambda x: .7*(x**2-1), lambda x: np.sin(1.5*x)][r.integers(0, 4)](z[p]) for p in ps); s = (s-s.mean())/s.std()
            k = r.integers(0, 6); m = 1/(1+np.exp(-1.2*s))
            if k == 0: c = r.choice([8, 15, 40]); x = np.clip(r.beta(c*m, c*(1-m)), 1e-4, 1-1e-4)
            elif k == 1: N = r.choice([20, 100, 288]); x = r.binomial(N, m)/N
            elif k == 2: x = r.poisson(np.exp(1.5+.8*s))*1.
            elif k == 3: x = r.negative_binomial(3, 3/(3+np.exp(2+.7*s)))*1.
            elif k == 4: x = np.exp(.8*s + .35*r.normal(size=n))
            else: x = s + .4*r.normal(size=n)*np.exp(.5*s)
        data[v] = x; truth[v] = ps; z[v] = (x-x.mean())/x.std()
    p = r.permutation(size); nm = {f"v{i}": f"x{p[i]}" for i in range(size)}
    return {nm[k]: x for k, x in data.items()}, {nm[k]: [nm[q] for q in ps] for k, ps in truth.items()}
