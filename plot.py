import matplotlib.pyplot as plt

# Plot for num_blocks = 10 and variable b

b_values = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45]
our_values = [.236, .53, 2.29, 8.053, 12.26, 50.502, 78.9, 155.223, 401.54, 845.946]
cvx_values = [.0047, .018, .13, .636, 1.06, 7.430, 20.45, 67.638, 120.63, 246.458]

plt.title('Time to Convergence vs. Block Size (Number of Blocks = 10)')
plt.semilogy(b_values, our_values, label='Alternating Projections')
plt.semilogy(b_values, cvx_values, label='SCS')
plt.legend()
plt.grid()
plt.xlabel('Block Size')
plt.ylabel('Time to Convergence (seconds)')
plt.savefig('figures/ap_fixed_num_blocks.pdf', bbox_inches='tight')
plt.close()

# Plot for b = 10 and varible num_blocks

nb_values = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
our_values = [.25, .705, 1.914, 5.84, 7.58, 8.00, 15.18, 20.64, 32.786, 68.091, 75.39, 211.16]
cvx_values = [.0122, 0.2, .129, .455, 1.196, 4.48, 19.95, 60.50, 173.271, 335.631, 590.820]

plt.title('Time to Convergence vs. Number of Blocks (Block Size = 10)')
plt.semilogy(nb_values + [75], our_values, label='Alternating Projections')
plt.semilogy(nb_values, cvx_values, label='SCS')
plt.legend()
plt.grid()
plt.xlabel('Number of Blocks')
plt.ylabel('Time to Convergence (seconds)')
plt.savefig('figures/ap_fixed_block_size.pdf', bbox_inches='tight')
plt.close()

nb_values = [10, 15, 20, 25, 30, 35, 40, 45, 50]
our_values = [
  0.471,
  0.658,
  0.914,
  1.069,
  1.507,
  1.908,
  2.023,
  2.372,
  3.082,
  6.032,
  8.969,
  17.699
]
cvx_values = [
  0.964,
  1.984,
  3.42,
  5.749,
  8.831,
  18.151,
  30.433,
  63.256,
  93.908
]

cvxopt_values = [
  0.685,
  1.185,
  2.217,
  2.778,
  4.552,
  6.42,
  8.07,
  10.365,
  13.781,
  21.989,
  45.43,
  112.952
]

plt.title('Time to Convergence vs. Number of Blocks (Block Size = 10)')
plt.semilogy(nb_values + [60, 75, 100], our_values, label='Interior Point Projected GD')
plt.semilogy(nb_values, cvx_values, label='SCS')
plt.semilogy(nb_values + [60, 75, 100], cvxopt_values, label='CVXOPT')
plt.legend()
plt.grid()
plt.xlabel('Number of Blocks')
plt.ylabel('Time to Convergence (seconds)')
plt.savefig('figures/ippgd_fixed_block_size.pdf', bbox_inches='tight')
plt.close()

b_values = [10, 15, 20, 25, 30, 35, 40, 45, 50]
our_values = [
  0.514,
  0.689,
  0.906,
  1.329,
  1.691,
  2.471,
  3.208,
  4.209,
  5.41
]

cvx_values = [
  0.884,
  2.142,
  3.882,
  8.557,
  11.057,
  18.957,
  29.727,
  46.016,
  66.673
]

cvxopt_values = [
  0.776,
  1.507,
  2.546,
  3.696,
  5.264,
  7.855,
  10.154,
  13.523,
  18.93
]

plt.title('Time to Convergence vs. Block Size (Number of Blocks = 10)')
plt.semilogy(b_values, our_values, label='Interior Point Projected GD')
plt.semilogy(b_values, cvx_values, label='SCS')
plt.semilogy(b_values, cvxopt_values, label='CVXOPT')
plt.legend()
plt.grid()
plt.xlabel('Block Size')
plt.ylabel('Time to Convergence (seconds)')
plt.savefig('figures/ippgd_fixed_num_blocks.pdf', bbox_inches='tight')
plt.close()
