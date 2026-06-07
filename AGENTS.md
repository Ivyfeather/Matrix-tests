# MATRIX TESTS KNOWLEDGE BASE

## OVERVIEW
Collection of RISC-V matrix operation test cases and compilation tools for processor verification workloads.

## STRUCTURE
Matrix-tests/
├── isa/                    # RISC-V matrix C test programs
│   ├── xiangshan_mul_full.c # Full matrix multiplication test
│   ├── load_store.c        # Basic memory access tests
│   └── generated_matrix_config.h # Parameters from configuration.ini
├── out/                    # Compiled RISC-V binaries (.out)
├── sim/                    # Simulation and trace processing
│   └── filter_trace.py     # Cleans and formats memory access traces
├── compiler.sh             # LLVM/Clang to RISC-V compilation script
├── test.ld                 # Linker script for RISC-V targets
└── a-crt.o / a-syscalls.o  # Runtime and syscall support objects

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add matrix test | `isa/` | Write C code using matrix intrinsics |
| Compile test | `./compiler.sh isa/your_test.c` | Generates binary in `out/` |
| Trace filtering | `sim/filter_trace.py` | Post-process NEMU traces |
| Memory layout | `test.ld` | Defines stack and data sections |

## CONVENTIONS
- **C Code**: Use matrix extension intrinsics (RV64GV) for hardware acceleration.
- **Compilation**: Use `compiler.sh` to ensure correct LLVM flags and linking.
- **Binaries**: All compiled outputs reside in `out/` with `.out` extension.
- **Configuration**: Include `generated_matrix_config.h` for matrix dimensions.

## ANTI-PATTERNS
- Do NOT hardcode matrix sizes, include `generated_matrix_config.h` instead.
- Do NOT run tests directly on host, they require RISC-V emulator or RTL.
- Do NOT modify `a-crt.o` or `a-syscalls.o` unless changing runtime boot logic.
- Do NOT skip the `out/` directory when checking for compiled artifacts.
