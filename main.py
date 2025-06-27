import pandas as pd
from src.data_loader import load_and_clean_data
from src.dsga import DSGA
from src.individual import Individual
from src.utils import comparison_map, rule_decoder
import pickle

cleaned_df = load_and_clean_data("dow_jones_index.data")

train_stock_data = cleaned_df[cleaned_df['quarter'] == 1]
test_stock_data = cleaned_df[(cleaned_df['quarter'] == 2) | (cleaned_df['date'] == '2011-03-25')]
test_stock_data = test_stock_data.sort_values('date')

pred_test_stock_data = test_stock_data[test_stock_data['date'] != '2011-03-25']
return_test_stock_data = test_stock_data[test_stock_data['date'] != '2011-06-24']
pred_weeks = sorted(pred_test_stock_data['date'].unique())

dsga = DSGA(N=25, NG=80, M=0.01, IS=0.05, SD=0.01, RLC=20, CL=2, W=10, stock_data=train_stock_data)
classification_rules = dsga.dsga()

decoded_rules = [rule_decoder(rule.individual) for rule in classification_rules]

buy_signals = []
week_map = {pred_weeks[i+1]: pred_weeks[i] for i in range(len(pred_weeks) - 1)}

for prediction_week, feature_week in week_map.items():
    feature_rows = pred_test_stock_data[pred_test_stock_data['date'] == feature_week]

    for _, row in feature_rows.iterrows():
        for rule_function in decoded_rules:
            if rule_function and rule_function(row):
                buy_signals.append((prediction_week, row['stock'], feature_week))
                break

signals_by_week = {}
for pred_week, stock, feature_week in buy_signals:
    if pred_week not in signals_by_week:
        signals_by_week[pred_week] = []
    signals_by_week[pred_week].append((stock, feature_week))

best_per_week = []
for pred_week, feature_list in signals_by_week.items():
    best_stock = None
    best_change = float('-inf')

    for stock, feature_week in feature_list:
        row = pred_test_stock_data[
            (pred_test_stock_data['stock'] == stock) &
            (pred_test_stock_data['date'] == feature_week)
            ]
        change = row.iloc[0]['percent_change_price']
        if change > best_change:
            best_stock = stock
            best_change = change

    best_per_week.append((pred_week, best_stock))

results = []
for week, stock in best_per_week:
    row = test_stock_data[
        (test_stock_data['stock'] == stock) &
        (test_stock_data['date'] == week)
    ]
    weekly_return = row.iloc[0]['percent_change_next_weeks_price']
    results.append((stock, week, weekly_return))

results_df = pd.DataFrame(results, columns=['stock', 'week', 'weekly_return'])
print(results_df)

quarterly_ror = results_df['weekly_return'].sum()
print(f'Rate of return for Q2: {quarterly_ror:.2f}%')

results_df.to_csv("results/q2_stock_results.csv", index=False)
with open("results/q2_summary.txt", "w") as f:
    f.write(f"Rate of return for Q2: {quarterly_ror:.2f}%\n")

bit_rules = [rule.individual for rule in classification_rules]
with open("results/classification_rules.pkl", "wb") as f:
    pickle.dump(bit_rules, f)

buy_signal_df = pd.DataFrame(buy_signals, columns=['prediction_week', 'stock', 'feature_week'])
buy_signal_df.to_csv("results/buy_signals.csv", index=False)
