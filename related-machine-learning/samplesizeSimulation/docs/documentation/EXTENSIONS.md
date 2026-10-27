# Image and momentum extensions

This is a feasibility assessment of the local implementation, dated
2026-10-13. No image benchmark, market-data backtest, stock recommendation,
or demonstrated profitability is claimed.

## What transfers

The useful abstraction is a learning curve: vary training sample size, fit a
model, evaluate on observations the model has not seen, and assess whether a
predefined performance criterion is met. The current package supplies tabular
learners, simulation loops, aggregate metrics, configuration keys, and a basic
chronological evaluator. It does not detect a universal sufficient sample size,
fit a raw-image network, or implement an online trading policy.

| Application | Reusable parts | Work still needed |
| --- | --- | --- |
| Image classification | Sample-size grid, repeated fitting, binary metrics, result tables | Image/embedding adapter, class-balanced subsampling, independent grouped test data, multiclass metrics |
| Raw-image CNN training | Learning-curve design and reporting | Training callbacks, tensor data loaders, GPU/runtime integration, architecture and optimizer controls |
| Momentum forecasting | Tabular model registry, chronological evaluation, learning-curve reporting | Causal features and labels, purged/gapped splits, regime analysis, execution-aware backtest |
| Online buy/sell decisions | A possible eligibility gate based on validated historical evidence | Forecast uncertainty, calibrated action rule, sequential monitoring, portfolio constraints, order/execution system |

The relative efficiency threshold presently compares observed performance to a
finite reference. A curve can flatten at poor performance, and a large sample
does not establish useful signal. Use an absolute performance target and a
relevant baseline as well as a stability criterion when extending it.

## Images

The fastest prototype uses frozen, pretrained image embeddings as tabular
features. A conventional classifier then produces learning curves. Training a
CNN directly needs a learner adapter instead of generating multivariate-normal
pixels. This is an architectural recommendation inferred from the package's
tabular interface. CNN learning-curve estimation is studied directly in
[Didyk et al., Strategies and impact of learning curve estimation for
CNN-based image classification](https://arxiv.org/abs/2310.08470).

A credible experiment would:

1. Define the task, class prevalence, deployment population, and an absolute
   target before looking at the final test set.
2. Split by the independent unit: patient, subject, camera, acquisition session,
   or original image. Augmentations of one image stay with their source.
3. Keep a final test set untouched. Use only development data to select the
   model and required sample size. Fit preprocessing and any learned embedding
   inside the appropriate training split; an externally frozen encoder can be
   reused if that choice is fixed beforehand.
4. Train on increasing numbers of unique labeled images, with controlled class
   balance and multiple random subsamples. Count images/groups per class, not
   augmented copies, and hold training-budget rules consistent across sizes.
5. Compare against a simple baseline, report uncertainty using independent
   groups, and require the target to hold across several larger sample sizes.
6. Validate the selected sample size and model once on the final test set,
   including a new acquisition source when deployment will face such shifts.

The current `nnet` model is a small feed-forward network for tabular data.
Binary accuracy/AUC do not cover multiclass classification, segmentation, or
object detection. Those tasks need probability-matrix outputs and appropriate
metrics such as macro F1, Dice/IoU, or mean average precision. Per-class learning
curves can reveal requirements hidden by overall accuracy.

## Momentum and sample-size eligibility

The proposed sequence is plausible as research: wait until sufficient usable
history is available, estimate future return/direction, and use a separate
decision rule to buy, exit, or hold. The sufficient-history gate must be learned
from past data and independently validated. It cannot be inferred from row count
or the present 0.95 relative efficiency threshold alone.

Forecast evaluation must preserve chronology; rolling-origin evaluation trains
using only earlier observations. This principle is described in
[Hyndman and Athanasopoulos, time-series cross-validation](https://otexts.com/fpp3/toolbox.html#time-series-cross-validation).
The existing ordered evaluator is a starting point, but its contiguous blocks
do not purge overlapping forward-return labels or enforce an execution delay.

An initial research design would use one preselected stock or a fixed universe,
one decision frequency, and one forward-return horizon. Features might include
lagged returns, a fixed momentum lookback, trailing volatility, and volume.
Every feature must be available at decision time; corporate actions, delisted
stocks, and changing index membership need point-in-time handling. A label for
time `t` must have matured before it is included in a later training set.

Use original chronological rows and expanding or rolling training windows.
Purge training labels that overlap validation periods and set a gap based on
the forecast horizon and information availability. Do not bootstrap individual
price rows, sort the result, and treat it as an independent time-series test.
Repeated or overlapping returns are dependent observations; raw sample count
is not an effective count of independent evidence. More old history can also
hurt when the market regime changes.

Select the model, lookbacks, thresholds, and candidate window sizes in an inner
historical validation loop. Assess the chosen complete policy on later outer
windows and an untouched final period. Repeatedly searching strategies on the
same backtest creates selection bias; this problem is formalized in
[Bailey et al., The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).
Sequentially checking the same threshold also needs an appropriate stopping
rule or a fixed review schedule; the package's simulation percentiles do not
supply that inference.

A proposed long-only decision rule, to be calibrated and tested:

```text
if usable_matured_training_labels < validated_required_history:
    HOLD / remain in cash
else:
    fit using only information available at the decision timestamp
    forecast return over the preselected horizon
    if uncertainty-adjusted forecast exceeds estimated round-trip costs + buffer:
        BUY or retain a bounded long position
    elif a preselected exit rule is met:
        SELL an existing long position
    else:
        HOLD / remain in cash
```

This is a proposed policy structure, not an implemented or validated signal.
Selling an existing holding is distinct from opening a short position, which
adds borrow availability, fees, and separate risk assumptions. The uncertainty
adjustment must be calibrated on training/development history; an AUC confidence
bound does not translate directly into a return forecast bound.

The backtest should simulate attainable next-period execution and include
spread, commissions, slippage, turnover, and any financing/borrow costs.
Execution quality changes transaction cost and realized price, as described in
the [SEC's Trade Execution guidance](https://www.sec.gov/about/reports-publications/investorpubstradexec).
Report net returns, drawdown, exposure, turnover, and performance against cash,
buy-and-hold, and a simple fixed momentum baseline. Directional accuracy alone
is insufficient: a few large losses can outweigh many correct directions.
Test robustness across periods and cost assumptions, then paper trade before
considering deployment. Required history should be revisited after drift under
a predeclared monitoring policy, not adjusted to rescue an unfavorable backtest.

## Recommended implementation boundary

Keep the first CRAN release focused on sample-size simulation for tabular
models. Build extensions around explicit data/learner/split/metric adapters:

```text
source data -> independent/chronological split -> training-size sampler
            -> fold-local fit/preprocessing -> predictions -> task metric
            -> uncertainty and target criterion -> eligibility decision
```

Add a real-data learning-curve API with a fixed external evaluation set, grouped
and purged/gapped splitters, and callbacks for image training. A trading module
would then consume its eligibility output plus an external walk-forward
forecast and execution backtest. This design makes both extensions reviewable
without implying that simulation convergence proves trading profitability.
