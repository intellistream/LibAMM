# LibAMM CI

LibAMM is the DataSys benchmark suite for approximate matrix multiplication.
Framework-neutral algorithm implementations and the compatibility Python package
are maintained in [AMM-Algorithms](https://github.com/DataSysResearch/AMM-Algorithms).

`benchmark-validation.yml` checks the maintained dataset tool, shell setup script,
required benchmark inventory, and CSV configuration readability on every pull
request and push to `main`.

This repository does not build or publish the `isage-amms` package. Package builds
and releases belong to AMM-Algorithms.
