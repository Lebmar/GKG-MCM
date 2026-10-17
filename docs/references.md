# Verified sources and claim boundaries

Checked on 17 October 2026. These are sources for the competition study, not
evidence that the proposed strategy will outperform. The competition PDF and
supplied synthetic CSV govern the experiment; external historical observations
must not replace the supplied evaluation paths.

## Data provenance and attribution

**Jordà, Ò., Schularick, M., and Taylor, A. M. (2017).** “Macrofinancial History
and the New Business Cycle Facts.” In *NBER Macroeconomics Annual 2016*, volume
31, edited by M. Eichenbaum and J. A. Parker. University of Chicago Press.

**Jordà, Ò., Knoll, K., Kuvshinov, D., Schularick, M., and Taylor, A. M. (2019).**
“The Rate of Return on Everything, 1870–2015.” *The Quarterly Journal of
Economics*, 134(3), 1225–1298. [DOI: 10.1093/qje/qjz012](https://doi.org/10.1093/qje/qjz012).

- Primary records read: [JST database and citation instructions](https://www.macrohistory.net/database/),
  [official licence](https://www.macrohistory.net/database/licence-terms/),
  [official R6 documentation, July 2022](https://www.macrohistory.net/app/download/9918957869/JST_RORE_Documentation_R6.pdf?t=1658254098),
  [2019 paper at the Federal Reserve Bank of San Francisco](https://www.frbsf.org/economic-research/wp-content/uploads/sites/4/wp2017-25.pdf).
- Supported use: the original database covers nominal local-currency returns;
  R6 extends return series through 2020. The database owner asks for the 2017
  citation generally and the 2019 citation for return data. Its website states
  **CC BY-NC-SA 4.0** and asks users to record the consultation date.
- Scope boundary: the competition CSV is synthetic, according to the problem;
  it is not the JST historical database itself. State that distinction and
  retain the supplied attribution. The raw historical database has not been
  downloaded or added as a new training source. This source does not validate
  our loss thresholds, client suitability policy, or predictive assumptions.

## Method sources

### Cost-aware sequential portfolio decisions

**Boyd, S., Busseti, E., Diamond, S., Kahn, R. N., Koh, K., Nystrup, P., and
Speth, J. (2017).** “Multi-Period Trading via Convex Optimization.”
*Foundations and Trends in Optimization*, 3(1), 1–76.
[DOI: 10.1561/2400000023](https://doi.org/10.1561/2400000023).

- Primary source read: [authors’ Stanford page](https://stanford.edu/~boyd/papers/cvx_portfolio.html)
  and [final manuscript](https://stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf),
  especially sections 2.3–2.5, 4.3, and 5.3.
- Supported use: portfolios can be evaluated with explicit self-financing
  accounting and a trade-off among expected return, risk, and transaction cost.
  Planned future trades are reconsidered as new observations become available.
- Scope boundary: this does not establish the value of our forecasts, make our
  drawdown objective convex, or prove optimality of a discrete candidate rule.
  Its generic trading fees are not the contest’s one-transfer fee convention.
  We implement the contest convention directly.
- Metadata note: the author page dates publication August 2017; the final PDF
  carries a 2016 volume designation and 2017 copyright. We use 2017 consistently
  with the author page and retain the verified DOI.

### Equal-weight comparison and estimation uncertainty

**DeMiguel, V., Garlappi, L., and Uppal, R. (2009).** “Optimal Versus Naive
Diversification: How Inefficient is the 1/N Portfolio Strategy?” *The Review
of Financial Studies*, 22(5), 1915–1953.
[DOI: 10.1093/rfs/hhm075](https://doi.org/10.1093/rfs/hhm075).

- Primary source read: [publisher record and abstract](https://academic.oup.com/rfs/article-abstract/22/5/1915/1592901).
  Publisher abstract is accessible; full publisher text was not accessed.
- Supported use: their out-of-sample study motivates retaining a simple 1/N
  comparator and taking estimation error seriously. Its abstract reports that
  none of 14 assessed models consistently surpassed 1/N across seven datasets
  on its chosen metrics.
- Scope boundary: those findings do not prove that equal weights are optimal
  for this three-asset synthetic task, nor that all dynamic strategies fail.
  The competition independently requires the annual equal-weight benchmark.
  Cite the 2009 issue year; the publisher also lists online publication in 2007.

### Peak-to-trough loss

**Magdon-Ismail, M., Atiya, A. F., Pratap, A., and Abu-Mostafa, Y. S. (2004).**
“On the Maximum Drawdown of a Brownian Motion.” *Journal of Applied
Probability*, 41(1), 147–161.
[DOI: 10.1239/jap/1077134674](https://doi.org/10.1239/jap/1077134674).

- Primary sources read: [publisher record](https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/on-the-maximum-drawdown-of-a-brownian-motion/F9E3B8A454B020DDEBF0AC3390EF7807)
  and [author-hosted paper, page 147](https://work.caltech.edu/pub/Magdon-Ismail2004drawdown.pdf).
- Supported use: drawdown represents loss from an earlier running peak to a
  subsequent trough. The paper formally studies absolute drawdown of an
  arithmetic Brownian process.
- Scope boundary: our implementation explicitly defines **relative drawdown of
  annual, inflation-adjusted net wealth**. That normalization is our modelling
  definition, not the Brownian formula in this source. No Brownian assumption
  or analytic Brownian tail probability is imported into the simulation.

## Claims requiring our own evidence

The references above do not establish the chosen 5% weight grid, utility risk
coefficients, liquidity floors, 10% empirical loss-exceedance cutoff, human
review rule, or superiority of any policy. Those must be labelled as declared
modelling choices, justified with development/validation evidence, and checked
by sensitivity analysis. Final-scenario outcomes may evaluate frozen choices
but must not choose or revise them.

The four analytical papers used in the methods/data discussion are the 2019
return-data paper, Boyd et al., DeMiguel et al., and Magdon-Ismail et al. The
2017 JST citation is retained additionally to satisfy the database owner's
general attribution instructions. No reference metadata has been guessed.

## Packaging the data licence

The [Creative Commons legal text, section 3(a)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt)
allows supplying the licence text **or** a licence link. For an offline contest
ZIP, a practical choice is to include the unchanged official plain-text file
as `LICENSE-DATA-CC-BY-NC-SA-4.0.txt`, alongside a separate data notice naming
the supplied synthetic file, its competition source, the JST attribution, and
whether this project changed the supplied file. This preserves attribution
without suggesting that the database authors endorse the project. Changes to
data should be identified; do not describe a modified file as unchanged.

The legal text itself is dedicated to the public domain under CC0 in its
closing notice, so an unchanged copy can be packaged. The [official summary](https://creativecommons.org/licenses/by-nc-sa/4.0/)
is useful for readers but does not replace the legal text. Keep the data
licence scope explicit rather than silently treating it as a licence for all
independently written source code. This note records a packaging practice
based on the official text; it does not resolve rights beyond the supplied
notices or change the competition's submission rules.
