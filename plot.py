import matplotlib.pyplot as plt

# Plot for num_blocks = 10 and variable b

b_values = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45]
our_values = [.236, .53, 2.29, 8.053, 12.26, 50.502, 78.9, 155.223, 401.54, 845.946]
cvx_values = [.0047, .018, .13, .636, 1.06, 7.430, 20.45, 67.638, 120.63, 246.458]

plt.title('Time to Convergence vs. Block Size (Number of Blocks = 10)')
plt.semilogy(b_values, our_values, label='LEigOpt')
plt.semilogy(b_values, cvx_values, label='CVXPY')
plt.legend()
plt.grid()
plt.xlabel('Block Size')
plt.ylabel('Time to Convergence (seconds)')
plt.savefig('figures/fixed_num_blocks.pdf')
plt.close()

# Plot for b = 10 and varible num_blocks

nb_values = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
our_values = [.25, .705, 1.914, 5.84, 7.58, 8.00, 15.18, 20.64, 32.786, 68.091, 75.39, 211.16]
cvx_values = [.0122, 0.2, .129, .455, 1.196, 4.48, 19.95, 60.50, 173.271, 335.631, 590.820]

plt.title('Time to Convergence vs. Number of Blocks (Block Size = 10)')
plt.semilogy(nb_values + [75], our_values, label='LEigOpt')
plt.semilogy(nb_values, cvx_values, label='CVXPY')
plt.legend()
plt.grid()
plt.xlabel('Number of Blocks')
plt.ylabel('Time to Convergence (seconds)')
plt.savefig('figures/fixed_block_size.pdf')