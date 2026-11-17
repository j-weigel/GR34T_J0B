window.NOTEBOOKS = window.NOTEBOOKS || {};
window.NOTEBOOKS["boids-swarming"] = {
 "name": "boids_swarming_EC.ipynb",
 "kind": "notebook",
 "download": "assets/notebooks/boids_swarming_EC.ipynb",
 "cells": [
  {
   "type": "markdown",
   "source": "# Boids: Swarming as Emergence (and Evolving It)\n**DCS 340 \u00b7 Project II starter notebook** \u00b7 based on Reynolds (1987), *Flocks, Herds, and Schools: A Distributed Behavioral Model*\n\n> \u26a0\ufe0f Scaffold notebook. The `\ud83d\udcdd Your notes` cells are for **you**: your predictions, observations, and connections to class. Those (and your blog/README) need to be in your own words per the GenAI policy.\n\n**Map of the notebook**\n1. The model: three local rules \u2192 one global behavior\n2. Watching it run (animation)\n3. Measuring emergence (polarization, spacing, group count)\n4. \ud83e\uddea Class activity A: *Predict, then run* (rule knockouts)\n5. \ud83e\uddea Class activity B: sliders / phase sweep\n6. Bringing in Evolutionary Computing: a GA that evolves the rule weights\n7. Bridge to Lehman & Stanley: objective vs. novelty search\n8. \ud83e\uddea Extension challenges (predator, field of view, topological neighbors)"
  },
  {
   "type": "code",
   "source": "# If running on Colab, everything below is preinstalled.\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom matplotlib import animation, rcParams\nfrom IPython.display import HTML, display\nfrom scipy.sparse.csgraph import connected_components\n\nrcParams['animation.embed_limit'] = 60  # MB, for longer animations\nrng_global = np.random.default_rng(340)",
   "outputs": []
  },
  {
   "type": "markdown",
   "source": "## 1. The model\n\nEach boid only sees neighbors inside a **perception radius** (and optionally a field of view). Reynolds' three behaviors, in his priority order:\n\n| Reynolds' name (1987) | Common modern name | Steers toward\u2026 |\n|---|---|---|\n| Collision avoidance | **Separation** | away from boids that are too close |\n| Velocity matching | **Alignment** | the average velocity of neighbors |\n| Flock centering | **Cohesion** | the average position of neighbors |\n\n**Steering, Reynolds-style:** each behavior produces a *desired velocity*; the steering force is `desired \u2212 current velocity`, capped at `max_force`.\n\n**Two ways to combine the three forces** (a detail worth knowing for reproducibility):\n- `mode=\"weighted\"`: weighted sum, which is what most tutorials and libraries do.\n- `mode=\"priority\"`: Reynolds' *prioritized acceleration allocation*: fill an acceleration \"budget\" in priority order (separation first), so in a crowd, avoidance can crowd out the others.\n\n\ud83d\udccc The world is a **torus** (wraps at the edges) to avoid wall effects."
  },
  {
   "type": "code",
   "source": "DEFAULTS = dict(\n    n=80, size=100.0,\n    radius=10.0,          # perception radius for alignment & cohesion\n    sep_radius=4.0,       # \"too close\" radius for separation\n    w_sep=1.5, w_ali=1.0, w_coh=1.0,\n    max_speed=2.0, min_speed=0.5, max_force=0.08,\n    fov_deg=360.0,        # field of view; 360 = sees all around\n    mode=\"weighted\",      # or \"priority\"\n)\n\ndef _limit(v, max_mag):\n    # Cap each row-vector's length at max_mag.\n    mag = np.linalg.norm(v, axis=1, keepdims=True)\n    scale = np.where(mag > max_mag, max_mag / np.maximum(mag, 1e-12), 1.0)\n    return v * scale\n\ndef _steer_toward(desired_dir, vel, p, has):\n    # Reynolds steering: desired velocity (at max speed) minus current velocity.\n    mag = np.linalg.norm(desired_dir, axis=1, keepdims=True)\n    desired = desired_dir / np.maximum(mag, 1e-12) * p[\"max_speed\"]\n    steer = _limit(desired - vel, p[\"max_force\"])\n    return np.where(has[:, None], steer, 0.0)\n\ndef init_flock(p, rng):\n    pos = rng.uniform(0, p[\"size\"], (p[\"n\"], 2))\n    ang = rng.uniform(0, 2*np.pi, p[\"n\"])\n    vel = np.c_[np.cos(ang), np.sin(ang)] * p[\"max_speed\"] * 0.75\n    return pos, vel\n\ndef pairwise(pos, size):\n    # diff[i, j] = vector from boid i to boid j, using torus wrap-around.\n    diff = pos[None, :, :] - pos[:, None, :]\n    diff = (diff + size/2) % size - size/2\n    dist = np.linalg.norm(diff, axis=2)\n    return diff, dist\n\ndef step(pos, vel, p):\n    diff, dist = pairwise(pos, p[\"size\"])\n    np.fill_diagonal(dist, np.inf)\n\n    visible = dist < p[\"radius\"]\n    if p[\"fov_deg\"] < 360:\n        heading = vel / np.linalg.norm(vel, axis=1, keepdims=True)\n        cosang = np.einsum('ik,ijk->ij', heading, diff) / np.maximum(dist, 1e-12)\n        visible &= cosang > np.cos(np.radians(p[\"fov_deg\"] / 2))\n    too_close = visible & (dist < p[\"sep_radius\"])\n\n    cnt = visible.sum(1)\n    has_n, has_close = cnt > 0, too_close.any(1)\n\n    # Separation: push away, weighted by 1/distance (closer = stronger)\n    away = -(diff / np.maximum(dist, 1e-12)[..., None]**2) * too_close[..., None]\n    sep = _steer_toward(away.sum(1), vel, p, has_close)\n    # Alignment: match average neighbor velocity\n    avg_vel = (vel[None, :, :] * visible[..., None]).sum(1) / np.maximum(cnt, 1)[:, None]\n    ali = _steer_toward(avg_vel, vel, p, has_n)\n    # Cohesion: head toward neighbors' center (offset from me)\n    center_off = (diff * visible[..., None]).sum(1) / np.maximum(cnt, 1)[:, None]\n    coh = _steer_toward(center_off, vel, p, has_n)\n\n    forces = [p[\"w_sep\"]*sep, p[\"w_ali\"]*ali, p[\"w_coh\"]*coh]\n    if p[\"mode\"] == \"weighted\":\n        acc = sum(forces)\n    else:  # prioritized acceleration allocation (Reynolds 1987)\n        acc = np.zeros_like(vel)\n        budget = np.full(len(vel), p[\"max_force\"] * 1.5)\n        for f in forces:  # priority order: separation, alignment, cohesion\n            fm = np.linalg.norm(f, axis=1)\n            take = np.minimum(fm, budget)\n            acc += f * (take / np.maximum(fm, 1e-12))[:, None]\n            budget -= take\n\n    vel = vel + acc\n    speed = np.linalg.norm(vel, axis=1, keepdims=True)\n    vel = vel / np.maximum(speed, 1e-12) * np.clip(speed, p[\"min_speed\"], p[\"max_speed\"])\n    pos = (pos + vel) % p[\"size\"]\n    return pos, vel\n\ndef simulate(steps=300, seed=0, record=True, **overrides):\n    p = {**DEFAULTS, **overrides}\n    rng = np.random.default_rng(seed)\n    pos, vel = init_flock(p, rng)\n    hist = []\n    for _ in range(steps):\n        pos, vel = step(pos, vel, p)\n        if record:\n            hist.append((pos.copy(), vel.copy()))\n    return (hist if record else (pos, vel)), p",
   "outputs": []
  },
  {
   "type": "markdown",
   "source": "## 2. Watching it run"
  },
  {
   "type": "code",
   "source": "def animate(hist, p, every=2, title=\"\"):\n    fig, ax = plt.subplots(figsize=(5, 5))\n    ax.set_xlim(0, p[\"size\"]); ax.set_ylim(0, p[\"size\"])\n    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')\n    ax.set_title(title)\n    pos, vel = hist[0]\n    q = ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1],\n                  angles='xy', scale_units='xy', scale=0.4, width=0.006, color='tab:blue')\n    def update(k):\n        pos, vel = hist[k * every]\n        q.set_offsets(pos); q.set_UVC(vel[:, 0], vel[:, 1])\n        return (q,)\n    anim = animation.FuncAnimation(fig, update, frames=len(hist)//every, interval=40, blit=True)\n    plt.close(fig)\n    return HTML(anim.to_jshtml())\n\nhist, p = simulate(steps=300, seed=1)\nanimate(hist, p, title=\"Default boids\")",
   "outputs": []
  },
  {
   "type": "markdown",
   "source": "\ud83d\udcdd **Your notes:** What did you expect before you ran it? What surprised you? Where is the \"leader\"? (There isn't one. How would you explain that to someone in one sentence?)"
  },
  {
   "type": "markdown",
   "source": "## 3. Measuring emergence\n\n\"It looks like a flock\" isn't a measurement. Three numbers that make the emergence quantitative:\n\n- **Polarization** $\\phi = \\left\\|\\frac{1}{N}\\sum_i \\hat v_i\\right\\|$: 0 = headings random, 1 = everyone moving the same direction. (This is the order parameter from Vicsek et al. 1995.)\n- **Mean nearest-neighbor distance (NND):** how tightly packed.\n- **Number of groups:** connected components when boids within a *fixed* link distance (6 units) are linked. (Fixed on purpose: an earlier version used each flock's own perception radius, and the GA below \"cheated\" by maxing out the radius so everything counted as one group.)"
  },
  {
   "type": "code",
   "source": "LINK_DIST = 6.0\n\ndef metrics(pos, vel, p):\n    heading = vel / np.linalg.norm(vel, axis=1, keepdims=True)\n    polar = np.linalg.norm(heading.mean(0))\n    _, dist = pairwise(pos, p[\"size\"])\n    np.fill_diagonal(dist, np.inf)\n    nnd = dist.min(1).mean()\n    n_groups, labels = connected_components(dist < LINK_DIST, directed=False)\n    largest = np.bincount(labels).max() / len(pos)\n    collisions = (dist < 1.0).sum() / 2      # pairs closer than 1 unit\n    return dict(polarization=polar, nnd=nnd, groups=n_groups,\n                largest_frac=largest, collisions=collisions)\n\ndef metric_traces(hist, p):\n    m = [metrics(pos, vel, p) for pos, vel in hist]\n    return {k: np.array([d[k] for d in m]) for k in m[0]}\n\ntr = metric_traces(hist, p)\nfig, axs = plt.subplots(1, 3, figsize=(13, 3))\nfor ax, k in zip(axs, [\"polarization\", \"nnd\", \"groups\"]):\n    ax.plot(tr[k]); ax.set_title(k); ax.set_xlabel(\"time step\")\nplt.tight_layout(); plt.show()",
   "outputs": []
  },
  {
   "type": "markdown",
   "source": "## 4. \ud83e\uddea Class activity A: *Predict, then run*\n\n**Instructions for the room:** for each mystery flock, write down a prediction *before* running: Will it align? Clump? Scatter? How many groups?\n\nThen run the cell and compare against the metrics."
  },
  {
   "type": "code",
   "source": "experiments = {\n    \"A. Separation only\":        dict(w_sep=1.5, w_ali=0.0, w_coh=0.0),\n    \"B. Alignment only\":         dict(w_sep=0.0, w_ali=1.0, w_coh=0.0),\n    \"C. Cohesion only\":          dict(w_sep=0.0, w_ali=0.0, w_coh=1.0),\n    \"D. No separation\":          dict(w_sep=0.0, w_ali=1.0, w_coh=1.0),\n    \"E. No alignment\":           dict(w_sep=1.5, w_ali=0.0, w_coh=1.0),\n    \"F. All three (baseline)\":   dict(),\n}\n\nfig, axs = plt.subplots(2, 3, figsize=(12, 8))\nfor ax, (name, ov) in zip(axs.flat, experiments.items()):\n    (pos, vel), p = simulate(steps=400, seed=2, record=False, **ov)\n    m = metrics(pos, vel, p)\n    ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], angles='xy',\n              scale_units='xy', scale=0.4, width=0.006)\n    ax.set_xlim(0, p[\"size\"]); ax.set_ylim(0, p[\"size\"]); ax.set_aspect('equal')\n    ax.set_xticks([]); ax.set_yticks([])\n    ax.set_title(f\"{name}\\n\u03c6={m['polarization']:.2f}  groups={m['groups']}  NND={m['nnd']:.1f}\", fontsize=9)\nplt.tight_layout(); plt.show()\n\n# Want to watch one? e.g.:\n# h, p = simulate(steps=300, seed=2, **experiments[\"D. No separation\"]); animate(h, p)",
   "outputs": []
  },
  {
   "type": "markdown",
   "source": "\ud83d\udcdd **Your notes / discussion prompts**\n- Which single rule is *necessary* for global alignment? Which one prevents collapse?\n- Alignment-only still produces order. What does that suggest about where \"flocking\" really comes from?\n- Is \"no separation\" realistic for real birds? Why did Reynolds put collision avoidance *first* in priority?"
  },
  {
   "type": "markdown",
   "source": "## 5. \ud83e\uddea Class activity B: sliders and a phase sweep\n\nRun the slider cell, then drag. **Challenge for groups:** find settings that produce (a) one tight ball, (b) several separate flocks, (c) a \"river\" (high polarization, spread out), (d) total chaos."
  },
  {
   "type": "code",
   "source": "try:\n    import ipywidgets as w\n    def explore(w_sep=1.5, w_ali=1.0, w_coh=1.0, radius=10.0, fov_deg=360, mode=\"weighted\"):\n        (pos, vel), p = simulate(steps=300, seed=3, record=False, w_sep=w_sep, w_ali=w_ali,\n                                 w_coh=w_coh, radius=radius, fov_deg=fov_deg, mode=mode)\n        m = metrics(pos, vel, p)\n        plt.figure(figsize=(4.5, 4.5))\n        plt.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], angles='xy', scale_units='xy', scale=0.4)\n        plt.xlim(0, p[\"size\"]); plt.ylim(0, p[\"size\"]); plt.gca().set_aspect('equal')\n        plt.title(\" \".join(f\"{k}={v:.2f}\" if isinstance(v, float) else f\"{k}={v}\" for k, v in m.items()), fontsize=8)\n        plt.show()\n    w.interact_manual(explore,\n        w_sep=(0.0, 4.0, 0.25), w_ali=(0.0, 4.0, 0.25), w_coh=(0.0, 4.0, 0.25),\n        radius=(2.0, 30.0, 1.0), fov_deg=(60, 360, 30), mode=[\"weighted\", \"priority\"])\nexcept ImportError:\n    print(\"ipywidgets not available: edit the parameters in explore(...) directly.\")",
   "outputs": []
  },
  {
   "type": "markdown",
   "source": "**Phase sweep:** how does global order depend on perception radius? (Local information \u2192 global structure.)"
  },
  {
   "type": "code",
   "source": "radii = [2, 4, 6, 8, 10, 14, 18, 25]\nseeds = range(3)\nphi = [[metric_traces(simulate(steps=250, seed=s, radius=r)[0][-50:], {**DEFAULTS, \"radius\": r})[\"polarization\"].mean()\n        for s in seeds] for r in radii]\nphi = np.array(phi)\nplt.errorbar(radii, phi.mean(1), yerr=phi.std(1), marker='o', capsize=3)\nplt.xlabel(\"perception radius\"); plt.ylabel(\"polarization (last 50 steps)\")\nplt.title(\"How much do boids need to see to flock?\"); plt.show()",
   "outputs": []
  },
  {
   "type": "markdown",
   "source": "\ud83d\udcdd **Your notes:** Is there a threshold? What's the analogy to a phase transition (cf. Vicsek model)?"
  },
  {
   "type": "markdown",
   "source": "## 6. Bringing in Evolutionary Computing\n\nReynolds hand-tuned his parameters. Can a **genetic algorithm** find them instead?\n\n- **Genome:** `[w_sep, w_ali, w_coh, radius]`\n- **Fitness (one possible objective; question it!):** high polarization + most boids in one group \u2212 collisions, averaged over the whole run (so flocks that form *quickly* score higher)\n- **Operators:** tournament selection, blend crossover, Gaussian mutation, elitism (same pieces as our earlier class GAs)"
  },
  {
   "type": "code",
   "source": "BOUNDS = np.array([[0, 4], [0, 4], [0, 4], [2, 25]])   # w_sep, w_ali, w_coh, radius\nGENE_NAMES = [\"w_sep\", \"w_ali\", \"w_coh\", \"radius\"]\n\ndef run_genome(g, seed, steps=150, n=50):\n    hist, p = simulate(steps=steps, seed=seed, n=n,\n                       w_sep=g[0], w_ali=g[1], w_coh=g[2], radius=g[3])\n    tr = metric_traces(hist[::5], p)   # average over the WHOLE run, so flocking *fast* is rewarded\n    return {k: v.mean() for k, v in tr.items()}\n\ndef objective(m):\n    return m[\"polarization\"] + m[\"largest_frac\"] - 0.05 * m[\"collisions\"]\n\ndef behavior(m):\n    # Behavior descriptor used later for novelty search\n    return np.array([m[\"polarization\"], m[\"largest_frac\"], m[\"nnd\"] / 10])\n\ndef ga(fitness_fn, pop_size=16, gens=12, seed=0, mut_sd=0.15, elite=2):\n    rng = np.random.default_rng(seed)\n    pop = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], (pop_size, 4))\n    log = []\n    for gen in range(gens):\n        ms = [run_genome(g, seed=gen) for g in pop]\n        fit = np.array(fitness_fn(ms, pop))\n        order = np.argsort(-fit)\n        log.append(dict(gen=gen, best=fit[order[0]], mean=fit.mean(),\n                        best_genome=pop[order[0]].copy(), metrics=ms, pop=pop.copy()))\n        print(f\"gen {gen:2d}  best={fit[order[0]]:.3f}  mean={fit.mean():.3f}  \"\n              f\"best genome={np.round(pop[order[0]], 2)}\")\n        new = [pop[i].copy() for i in order[:elite]]\n        while len(new) < pop_size:\n            a, b = [max(rng.choice(pop_size, 3), key=lambda i: fit[i]) for _ in range(2)]\n            alpha = rng.uniform(-0.25, 1.25, 4)                 # blend crossover\n            child = pop[a] + alpha * (pop[b] - pop[a])\n            child += rng.normal(0, mut_sd, 4) * (BOUNDS[:, 1] - BOUNDS[:, 0])\n            new.append(np.clip(child, BOUNDS[:, 0], BOUNDS[:, 1]))\n        pop = np.array(new)\n    return log\n\nobj_log = ga(lambda ms, pop: [objective(m) for m in ms])",
   "outputs": []
  },
  {
   "type": "code",
   "source": "plt.plot([l[\"best\"] for l in obj_log], label=\"best\")\nplt.plot([l[\"mean\"] for l in obj_log], label=\"mean\")\nplt.xlabel(\"generation\"); plt.ylabel(\"fitness\"); plt.legend(); plt.show()\n\nbest = obj_log[-1][\"best_genome\"]\nprint(dict(zip(GENE_NAMES, np.round(best, 2))))\nh, p = simulate(steps=300, seed=10, w_sep=best[0], w_ali=best[1], w_coh=best[2], radius=best[3])\nanimate(h, p, title=\"GA-evolved flock\")",
   "outputs": []
  },
  {
   "type": "markdown",
   "source": "\ud83d\udcdd **Your notes / discussion prompts**\n- Does the evolved flock *look* like a flock, or did the GA find a loophole in the fitness function? (Look closely at the genome. Is any weight near 0 or the max?)\n- What did the fitness function **reward** vs. what you **meant**? This is the core of the Lehman & Stanley reading.\n- Try changing `objective`, e.g. reward *multiple* groups, or penalize clumping (low NND)."
  },
  {
   "type": "markdown",
   "source": "## 7. Bridge to Lehman & Stanley: novelty search\n\nInstead of rewarding an objective, reward being **different** from what has been seen before. Each genome's *behavior* = (polarization, largest-group fraction, spacing). Novelty = mean distance to its *k* nearest neighbors among the current population + an archive."
  },
  {
   "type": "code",
   "source": "archive = []\ndef novelty_fitness(ms, pop, k=5):\n    B = np.array([behavior(m) for m in ms])\n    ref = np.vstack([B] + ([np.array(archive)] if archive else []))\n    d = np.linalg.norm(B[:, None] - ref[None], axis=2)\n    nov = np.sort(d, axis=1)[:, 1:k+1].mean(1)          # skip self (distance 0)\n    for i in np.argsort(-nov)[:2]:                       # add the 2 most novel to archive\n        archive.append(B[i])\n    return nov\n\nnov_log = ga(novelty_fitness)",
   "outputs": []
  },
  {
   "type": "code",
   "source": "def all_behaviors(log):\n    return np.array([behavior(m) for l in log for m in l[\"metrics\"]])\n\nBo, Bn = all_behaviors(obj_log), all_behaviors(nov_log)\nplt.figure(figsize=(6, 5))\nplt.scatter(Bo[:, 0], Bo[:, 2], s=12, alpha=.5, label=\"objective GA\")\nplt.scatter(Bn[:, 0], Bn[:, 2], s=12, alpha=.5, label=\"novelty search\")\nplt.xlabel(\"polarization\"); plt.ylabel(\"spacing (NND/10)\")\nplt.title(\"Which search explored more kinds of collective behavior?\"); plt.legend(); plt.show()",
   "outputs": []
  },
  {
   "type": "markdown",
   "source": "\ud83d\udcdd **Your notes:** Where did each search spend its time? Which would you use to *discover* collective behaviors nobody designed, vs. to *engineer* one you already want (e.g. a drone swarm)?"
  },
  {
   "type": "markdown",
   "source": "## 8. \ud83e\uddea Extension challenges (pick one in small groups)\n\n1. **Predator:** add a point that moves through the world; boids within some radius get a strong flee force (highest priority). Do you see the flock split and re-merge (\"fountain effect\")?\n2. **Field of view:** set `fov_deg=270` or `120`. Real birds have blind spots. How does it change polarization?\n3. **Topological neighbors:** Ballerini et al. (2008) found starlings track a fixed *number* of neighbors (~6\u20137), not everyone within a distance. Modify `step` so `visible` = each boid's 7 nearest. What changes when density changes?\n4. **Evolve something new:** add `fov_deg` or `max_force` to the genome."
  },
  {
   "type": "code",
   "source": "# Challenge 1 starter: predator.\n# Hint: compute the torus-wrapped vector from each boid to the predator,\n# make a flee steering force with _steer_toward(-to_pred, vel, p, near_mask),\n# and add it with a big weight (or put it first in the priority list).\n\ndef step_with_predator(pos, vel, p, pred_pos, flee_radius=15.0, w_flee=3.0):\n    pos, vel = step(pos, vel, p)        # normal boids update\n    # TODO: your code here\n    return pos, vel",
   "outputs": []
  },
  {
   "type": "code",
   "source": "# Challenge 3 starter: topological neighbors.\n# Hint: in step(), replace `visible = dist < p[\"radius\"]` with a mask that is True\n# for each row's k smallest distances, e.g. using np.argsort(dist, axis=1)[:, :k].",
   "outputs": []
  },
  {
   "type": "markdown",
   "source": "## References & further reading (verify before citing!)\n- Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model. *SIGGRAPH '87*. https://dl.acm.org/doi/10.1145/37402.37406\n- Reynolds' Boids page: https://www.red3d.com/cwr/boids/\n- Reynolds, C. W. (1999). Steering behaviors for autonomous characters. *GDC*.\n- Vicsek, T. et al. (1995). Novel type of phase transition in a system of self-driven particles. *Physical Review Letters*.\n- Couzin, I. D. et al. (2002). Collective memory and spatial sorting in animal groups. *Journal of Theoretical Biology*.\n- Ballerini, M. et al. (2008). Interaction ruling animal collective behavior depends on topological rather than metric distance. *PNAS*.\n- Kennedy, J. & Eberhart, R. (1995). Particle swarm optimization. *IEEE ICNN*.\n- Lehman, J. & Stanley, K. O. (2011). Abandoning objectives: Evolution through the search for novelty alone. *Evolutionary Computation*."
  }
 ]
};
