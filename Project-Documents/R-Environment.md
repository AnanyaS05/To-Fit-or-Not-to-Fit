# R Environment

This project's Bayesian pipeline was verified locally with:

- R `4.5.2`
- `brms` `2.23.0`
- `rstan` `2.32.7`
- `jsonlite` `2.0.0`

Windows note:

- Use a working Rtools toolchain.
- `Rscript` path used for the verified runs:
  `C:\Program Files\R\R-4.5.2\bin\Rscript.exe`

Install the required R packages:

```r
install.packages(c("brms", "rstan", "jsonlite"))
```

Quick verification:

```powershell
Rscript -e "library(rstan); library(brms); library(jsonlite); cat(as.character(getRversion()), '\n')"
```

Local `sessionInfo()` snapshot used for this submission package:

```text
R version 4.5.2 (2025-10-31 ucrt)
Platform: x86_64-w64-mingw32/x64
Running under: Windows 11 x64 (build 26200)

attached base packages:
[1] stats     graphics  grDevices utils     datasets  methods   base

other attached packages:
[1] jsonlite_2.0.0      rstan_2.32.7        StanHeaders_2.32.10
[4] brms_2.23.0         Rcpp_1.1.1
```
