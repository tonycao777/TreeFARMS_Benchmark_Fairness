from prettytable import PrettyTable
import numpy as np

def save_pretty_table(final_result, filename='final_results_table.txt'):
    """
    Creates a pretty table from the final_result dictionary and saves it to a text file.
    
    Args:
    - final_result: A dictionary containing summarized results for each dataset.
    - filename: The filename for the text output.
    """
    # Create a PrettyTable object
    table = PrettyTable()

    # Define the table columns
    table.field_names = ["Dataset", "Method", "Metric", "Mean", "Standard Deviation"]

    # Populate the table with data from final_result
    for dataset, methods in final_result.items():
        for method, metrics in methods.items():
            for metric, stats in metrics.items():
                if stats == []:
                    table.add_row([dataset, method, metric, np.nan, np.nan])
                else:
                    table.add_row([dataset, method, metric, stats["mean"], stats["std"]])

    # Save the table to a text file
    with open(filename, 'w') as f:
        f.write(str(table))
    
    print(f"Pretty table saved to {filename}")