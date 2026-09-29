/* ==========================================================================
   The shop — spend thoughts on upgrades (home page).

   Opened by clicking the Shop neuron below the fold. Each item can be bought
   up to `max` times; every purchase costs `growth` times more than the last.
   Levels are saved with the thought count (see Mind in cosmos.js), so they
   persist across visits.

   To add an upgrade: give it an `effect`, set `locked: false`, and apply the
   effect in applyEffects().

   Also home to the Reset neuron's two-step confirmation.
   ========================================================================== */
(function () {
  "use strict";

  const Mind = window.Mind;
  if (!Mind) return;

  const ITEMS = [
    {
      id: "omega3", name: "Omega-3", glyph: "◈",
      blurb: "Fatty acids for supple neural membranes. Each dose makes the network fire 10% more often, so thoughts come faster.",
      base: 200, growth: 1.25, max: 10, effect: 0.1,
    },
    { id: "caffeine", name: "Caffeine", glyph: "✧", locked: true },
    { id: "deep-sleep", name: "Deep Sleep", glyph: "☾", locked: true },
    { id: "myelination", name: "Myelination", glyph: "≋", locked: true },
    { id: "neuroplasticity", name: "Neuroplasticity", glyph: "❋", locked: true },
  ];

  const price = (item, level) => Math.round(item.base * Math.pow(item.growth, level));

  // Synapse activity: how often the network fires on its own.
  function applyEffects() {
    const omega = ITEMS[0];
    Mind.setActivity(1 + omega.effect * Mind.owned(omega.id));
  }
  applyEffects();

  // ---- Dialog -------------------------------------------------------------

  const dialog = document.createElement("dialog");
  dialog.className = "shop";
  dialog.setAttribute("aria-labelledby", "shop-title");
  dialog.innerHTML = `<div class="shop-panel">
    <header class="shop-head">
      <div>
        <div class="kicker">Spend your thoughts</div>
        <h2 id="shop-title">Shop</h2>
      </div>
      <button class="shop-close" type="button" aria-label="Close shop">×</button>
    </header>
    <p class="shop-balance">Thoughts: <strong></strong><span class="shop-mind-level"></span></p>
    <ul class="shop-items"></ul>
    <p class="shop-foot">Synapse activity: <strong></strong></p>
  </div>`;
  document.body.appendChild(dialog);

  const list = dialog.querySelector(".shop-items");
  const balance = dialog.querySelector(".shop-balance strong");
  const foot = dialog.querySelector(".shop-foot strong");

  list.innerHTML = ITEMS.map((item) => `
    <li class="shop-item${item.locked ? " locked" : ""}" data-id="${item.id}">
      <span class="shop-glyph" aria-hidden="true">${item.glyph}</span>
      <div class="shop-text">
        <h3>${item.name}${item.locked ? "" : ' <span class="shop-level"></span>'}</h3>
        <p>${item.locked ? "Locked" : item.blurb}</p>
      </div>
      ${item.locked
        ? '<span class="shop-lock" aria-label="Locked">🔒</span>'
        : '<button class="shop-buy" type="button"></button>'}
    </li>`).join("");

  function render() {
    if (!dialog.open) return;
    const have = Mind.thoughts;
    balance.textContent = have.toLocaleString();
    dialog.querySelector(".shop-mind-level").textContent = `Level ${Mind.level}`;
    for (const item of ITEMS) {
      if (item.locked) continue;
      const li = list.querySelector(`[data-id="${item.id}"]`);
      const level = Mind.owned(item.id);
      const btn = li.querySelector(".shop-buy");
      li.querySelector(".shop-level").textContent = `${level}/${item.max}`;
      if (level >= item.max) {
        btn.textContent = "Maxed";
        btn.disabled = true;
      } else {
        const cost = price(item, level);
        btn.innerHTML = `Buy <span>${cost.toLocaleString()}</span>`;
        btn.disabled = have < cost;
        btn.title = have < cost ? `You need ${(cost - have).toLocaleString()} more thoughts` : "";
      }
    }
    const omega = ITEMS[0];
    foot.textContent = `+${Math.round(omega.effect * Mind.owned(omega.id) * 100)}%`;
  }

  list.addEventListener("click", (e) => {
    const btn = e.target.closest(".shop-buy");
    if (!btn) return;
    const li = btn.closest(".shop-item");
    const item = ITEMS.find((i) => i.id === li.dataset.id);
    const level = Mind.owned(item.id);
    if (level >= item.max || !Mind.spend(price(item, level))) return;
    Mind.setOwned(item.id, level + 1);
    applyEffects();
    li.classList.remove("bought");
    void li.offsetWidth; // restart the purchase glow
    li.classList.add("bought");
    render();
  });

  dialog.querySelector(".shop-close").addEventListener("click", () => dialog.close());
  // Clicking the dimmed backdrop (outside the panel) closes it too.
  dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });
  Mind.onChange(render);

  // ---- Reset: open the prompt, then the button needs two clicks -----------

  const reset = document.createElement("dialog");
  reset.className = "shop reset";
  reset.setAttribute("aria-labelledby", "reset-title");
  reset.innerHTML = `<div class="shop-panel">
    <header class="shop-head">
      <div>
        <div class="kicker">Forget everything</div>
        <h2 id="reset-title">Reset</h2>
      </div>
      <button class="shop-close" type="button" aria-label="Cancel">×</button>
    </header>
    <p class="reset-text">This erases all your thoughts, your level, and everything you've bought, and returns the network to its starting neurons. It can't be undone.</p>
    <div class="reset-actions">
      <button class="reset-cancel" type="button">Keep my mind</button>
      <button class="reset-go" type="button">Reset</button>
    </div>
  </div>`;
  document.body.appendChild(reset);
  const go = reset.querySelector(".reset-go");
  let armed = false;
  const disarm = () => { armed = false; go.textContent = "Reset"; go.classList.remove("armed"); };
  go.addEventListener("click", () => {
    if (!armed) {
      armed = true;
      go.textContent = "Click again to confirm";
      go.classList.add("armed");
      return;
    }
    Mind.reset();
    applyEffects();
    reset.close();
  });
  reset.querySelector(".shop-close").addEventListener("click", () => reset.close());
  reset.querySelector(".reset-cancel").addEventListener("click", () => reset.close());
  reset.addEventListener("click", (e) => { if (e.target === reset) reset.close(); });
  reset.addEventListener("close", disarm);

  window.Shop = {
    open() {
      if (dialog.open) return;
      dialog.showModal();
      render();
    },
    confirmReset() {
      if (reset.open) return;
      disarm();
      reset.showModal();
      reset.querySelector(".reset-cancel").focus();
    },
  };
})();
