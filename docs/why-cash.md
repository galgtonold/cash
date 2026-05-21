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
