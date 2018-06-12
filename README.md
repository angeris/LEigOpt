# LEigOPT

This is a proof-of-concept, first-order interior point solver for eigenvalue optimization SDPs. It can
achieve high accuracies of around 6-8 significant digits while remaining
relatively fast, given a fixed sparsity pattern. For more information, see the
final report in the repository.

## Testing

In order to test this package, navigate to the `test.jl` file, under the `julia/`
folder and simply run `julia test.jl`.