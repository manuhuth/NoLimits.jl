# How to Contribute

Contributions and feedback are welcome - bug reports, feature requests, documentation
improvements, and code.

## Reporting issues and asking questions

- **Open an issue** on [GitHub](https://github.com/manuhuth/NoLimits.jl/issues) to report a
  bug, request a feature, or ask a question. For bug reports, please include a minimal
  reproducible example and the output of `versioninfo()` together with your NoLimits.jl
  version.
- **Get in touch** by emailing the lead developer and maintainer, Manuel Huth, at
  [manuel.huth@uni-bonn.de](mailto:manuel.huth@uni-bonn.de).

## Contributing code

1. Fork the repository and create a topic branch from `main`.
2. Set up the development environment and read the conventions in the
   [Developers Guide](developers-guide.md) - in particular the differentiability, formatting,
   and `get_*` accessor conventions.
3. Make your change with accompanying tests. New functionality should add or extend a file
   under `test/` and stay wired into `test/runtests.jl`.
4. Run the test suite locally before opening a pull request:
   ```bash
   julia --project -e 'using Pkg; Pkg.test()'
   ```
5. If your change affects user-facing behavior, update the relevant documentation page (and
   docstrings, which are rendered into the [API Reference](api.md)).
6. Open a pull request describing the change and the motivation. Continuous integration runs
   the test suite and builds the documentation on every pull request.

For larger features or design changes, opening an issue to discuss the approach first is
encouraged.
