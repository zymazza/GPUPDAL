---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

<!--
GitHub issues are for reproducible GPUPAL bugs and feature requests. Do not
include private point-cloud data, credentials, or unredacted environment
variables. Use the repository's private vulnerability-reporting channel for
security issues.
-->

**Describe the bug**
A clear and concise description of what the bug is, including steps to
reproduce it. Include the GPUPAL command (for example,
`gpupal translate ...`) and, when applicable, the PDAL-compatible pipeline.

```
$ gpupal translate in.las out.las
```

<details>
  
```
{
  "pipeline":[
    "in.las",
    "out.las"
  ]
}
```

</details>

**Expected behavior**
A clear and concise description of what you expected to happen.

**System/installation information:**
Please provide `gpupal version`, `gpupal doctor`, the sibling PDAL version, and
system information (for example, `uname -a`).

```
$ uname -a
```

```
$ gpupal version
$ gpupal doctor
```

If installed via Conda, you may be asked to paste the output of `conda list` and `conda info` as well.

<details>
  
```
$ conda list
```

</details>

<details>
  
```
$ conda info
```

</details>
