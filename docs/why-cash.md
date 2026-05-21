# Why Cash?

You probably aren't shopping for a notebook-caching library — there isn't one
to shop for. But there's a good chance you've already built one in pieces,
without calling it that. **Cash is the library you'd write yourself if you
had a spare month.**

## Does this sound familiar?

<div class="cash-pain-grid" markdown="0">

  <div class="cash-pain-card">
    <span class="cash-pain-icon">🕐</span>
    <span class="cash-pain-title">"Restart and Run All takes 20 minutes."</span>
    <span class="cash-pain-sub">Every small edit pays the full pipeline cost. You stop iterating and start avoiding restarts.</span>
  </div>

  <div class="cash-pain-card">
    <span class="cash-pain-icon">❓</span>
    <span class="cash-pain-title">"You're not sure if <code>df</code> is stale."</span>
    <span class="cash-pain-sub">Did the upstream cell change? Did someone mutate it? You squint at the timestamp and hope.</span>
  </div>

  <div class="cash-pain-card">
    <span class="cash-pain-icon">🥒</span>
    <span class="cash-pain-title">"Your notebook has 12 pickle files."</span>
    <span class="cash-pain-sub"><code>tmp.pkl</code>, <code>df_v3_USE_THIS.pkl</code>. You're afraid to delete any of them.</span>
  </div>

  <div class="cash-pain-card">
    <span class="cash-pain-icon">🌅</span>
    <span class="cash-pain-title">"Your kernel has been running for 4 days."</span>
    <span class="cash-pain-sub">You can't restart — that's where the state lives. One crash and you rebuild from scratch.</span>
  </div>

</div>

If two of those landed, keep reading.

## What happens when you turn cash on

Above each cell, cash shows a **badge** summarising what it did. Watch the same notebook through four states — first run, re-run, restart, then editing an upstream cell:

<iframe class="cash-badge" src="/_badges/why_cash_reel.html" loading="lazy" scrolling="no" height="260" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>
