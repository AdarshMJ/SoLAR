echo "Running SoLAR Deletions with hyperparameter search..."

# Define hyperparameter ranges
learning_rates=(0.0001 0.001 0.01)
hidden_dimensions=(32 64 128)
#max_iters_delete_values=(500 1000 3000 6000)

# Output file
output_file="hyperparameter_baseline.csv"

# Variables to keep track of the best result
best_accuracy=0
best_params=""

# Flag for first run
first_run=true

for oglr in "${learning_rates[@]}"; do
  for relr in "${learning_rates[@]}"; do
    for hidden_dim in "${hidden_dimensions[@]}"; do
      for max_delete in "${max_iters_delete_values[@]}"; do
        echo "Running with oglr=$oglr, relr=$relr, hidden_dim=$hidden_dim, max_delete=$max_delete"
        python splitrewiring.py \
          --dataset Cora \
          --out temp_result_baseline.csv \
          --oglr $oglr \
          --oghidden_dimension $hidden_dim \
          --remodel SimpleGCN \
          --relr $relr \
          --rehidden_dimension $hidden_dim \
          --max_iters_add 0 \
          --max_iters_delete $max_delete \
          --num_train 20 \
          --num_val 500
        
        if $first_run; then
          # For the first run, copy the entire file including headers
          cp temp_result_chameleon.csv $output_file
          first_run=false
        else
          # For subsequent runs, append only the data (skip header)
          tail -n +2 temp_result_chameleon.csv >> $output_file
        fi
        
        # Extract the validation accuracy from the temporary file (5th column)
        accuracy=$(tail -n 1 temp_result_chameleon.csv | cut -d',' -f5)
        
        # Update best result if current accuracy is higher
        if (( $(echo "$accuracy > $best_accuracy" | bc -l) )); then
          best_accuracy=$accuracy
          best_params="oglr=$oglr, relr=$relr, hidden_dim=$hidden_dim, max_delete=$max_delete"
        fi
        
        rm temp_result_chameleon.csv
      done
    done
  done
done

echo "Hyperparameter search completed. Results saved to $output_file"
echo "Best hyperparameters: $best_params"
echo "Best validation accuracy: $best_accuracy"

# Print the row with the best validation accuracy
echo "Full result row for best performance:"
grep "$best_accuracy" $output_file