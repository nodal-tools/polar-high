# About

polar-high-opt was created by **Juha Kiviluoma** of **Nodal-Tools** using
**Claude Opus**.

The code is heavily tested, but **correct functioning is not
guaranteed**. If you use it in production, run your own validation
against a reference model.

## Validation

The first downstream user is the
[FlexTool](https://github.com/irena-flextool/flextool) energy-system
modelling toolkit. FlexTool's test fleet — comparing the new
polar-high-opt build path against the earlier GNU MathProg one across
many scenarios — is the primary correctness validation for the
kernel.

## Reporting issues

Issue tracker:
[github.com/nodal-tools/polar-high-opt/issues](https://github.com/nodal-tools/polar-high-opt/issues).

Please include:

- the polar-high-opt version (`pip show polar-high-opt`),
- a minimal reproducer (the smaller, the faster the fix),
- expected vs observed `obj` and / or `Solution.highs.getModelStatus()`.

## License

Apache-2.0. See [License](license.md).
