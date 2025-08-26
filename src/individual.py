import numpy as np
import pandas as pd
from typing import List, Set, Tuple
from src.utils import comparison_map, sign_map

class Individual:
    """
    A single candidate solution encoded as a 43-bit rule.
    Bit layout:
      [0]     : buy-rule switch (1=enabled, 0=disabled → invert later)
      [1:11] : price change rule (on/off, relation (2b), sign(1b), magnitude(7b))
      [12:22]: volume change rule (on/off, relation (2b), sign(1b), magnitude(7b))
      [23:32]: days-to-next-dividend rule (on/off, relation (2b), magnitude(7b))
      [33:42]: next-dividend return rule (on/off, relation (2b), magnitude(7b))
    """
    def __init__(self) -> None:
        self.individual = self.random_init()

    def random_init(self) -> np.ndarray:
        return np.random.randint(0, 2, size=43)
    
    @staticmethod
    def distance(sw1: set[tuple[str, str]], sw2: set[tuple[str, str]]) -> float:
        overlap = len(sw1.intersection(sw2))
        if overlap > 0:
            distance = 1 / overlap
        else:
            return float("inf")
        return distance
    
    def fitness_function(self, tabu_list: list[tuple["Individual", set[tuple[str, str]]]], stock_data: pd.DataFrame, weight: int) -> None:
        sw_pairs = self.get_sw_pairs(stock_data)
        penalty_adjustment = 1
        length = len(sw_pairs)

        if length < weight:
            fitness = 0
        else:
            fitness = np.mean([
                stock_data.loc[(stock_data["stock"] == stock) & (stock_data["date"] == week), "percent_change_next_weeks_price"].values[0]
                for stock, week in sw_pairs
                ])
            
        if self.individual[0] == 0:  
            fitness *= -1

        # Diversity penalty against tabu individuals with similar selections.
        for stored_individual, stored_sw_pair in tabu_list:
            penalty_adjustment += 1 - ((self.distance(self.get_sw_pairs(stock_data), stored_sw_pair)) / 0.1)
        
        fitness += 10
        
        self.fitness = fitness
        self.shared_fitness = fitness / penalty_adjustment

    def get_sw_pairs(self, stock_data: pd.DataFrame) -> set[tuple[str, str]]:
        """
        Decode the chromosome into a set of (stock, date) pairs that pass all active rule filters (on/off).
        For each row in stock_data:
          - Optionally test price, volume, dividend-days, and dividend-return conditions.
          - If a condition is 'included' (bit==1), we apply the comparison operator to the feature and threshold.
          - Only rows passing all included conditions are selected.
        """
        sw_pairs = set()

        for index, row in stock_data.iterrows():
            stock, week = row["stock"], row["date"]

            buy_rule = self.individual[0]
            
            include_price_change = self.individual[1]
            price_change_relation = self.relational_symbol(self.individual[2:4])
            price_change_sign = self.sign(self.individual[4])
            price_change = self.bit_to_decimal(self.individual[5:12]) / 10
            
            include_vol_change = self.individual[12]
            vol_change_relation = self.relational_symbol(self.individual[13:15])
            vol_change_sign = self.sign(self.individual[15])
            vol_change = self.bit_to_decimal(self.individual[16:23])

            include_dividend_days = self.individual[23]
            dividend_days_relation = self.relational_symbol(self.individual[24:26])
            dividend_days = self.bit_to_decimal(self.individual[26:33])

            include_dividend_return = self.individual[33]
            dividend_return_relation = self.relational_symbol(self.individual[34:36])
            dividend_return = self.bit_to_decimal(self.individual[36:43]) / 100

            if include_price_change == 1:
                comp_function = comparison_map[price_change_relation]
                target = sign_map[price_change_sign] * price_change
                if not comp_function(row['percent_change_price'], target):
                    continue

            if include_vol_change == 1:
                if pd.isnull(row["percent_change_volume_over_last_wk"]):
                    continue
                comp_function = comparison_map[vol_change_relation]
                target = sign_map[vol_change_sign] * vol_change
                if not comp_function(row['percent_change_volume_over_last_wk'], target):
                    continue

            if include_dividend_days == 1:
                comp_function = comparison_map[dividend_days_relation]
                target = dividend_days
                if not comp_function(row['days_to_next_dividend'], target):
                    continue

            if include_dividend_return == 1:
                comp_function = comparison_map[dividend_return_relation]
                target = dividend_return
                if not comp_function(row['percent_return_next_dividend'], target):
                    continue

            sw_pairs.add((stock, week))

        return sw_pairs

    @staticmethod
    def bit_to_decimal(gene: np.ndarray) -> int:
        gene_str = ''.join(map(str, gene))
        reversed_gene = gene_str[::-1] # Little-endian form
        return int(reversed_gene, 2)
    
    @staticmethod
    def relational_symbol(gene: np.ndarray) -> str:
        gene_str = ''.join(map(str, gene))
        symbol_map = {"00": "<", "01": "<=", "10": ">", "11": ">="}
        return symbol_map.get(gene_str)
    
    @staticmethod
    def sign(bit: int) -> str:
        sign_map = {0: "-", 1: "+"}
        return sign_map.get(bit)

def rule_decoder(rule):
    """
    Convert a 43-bit individual into a Python function that, given a row, returns True if the row satisfies the rule.
    """
    if rule[0] == 0:
        return None
    
    rule_conditions = []

    if rule[1] == 1:
        relation = Individual.relational_symbol(rule[2:4])
        sign = Individual.sign(rule[4])
        value = Individual.bit_to_decimal(rule[5:12]) / 10
        if sign == '-':
            value *= -1
        comparison = comparison_map[relation]
        rule_conditions.append(lambda row, c=comparison, v=value: c(row['percent_change_price'], v))

    if rule[12] == 1:
        relation = Individual.relational_symbol(rule[13:15])
        sign = Individual.sign(rule[15])
        value = Individual.bit_to_decimal(rule[16:23])
        if sign == '-':
            value *= -1
        comparison = comparison_map[relation]
        rule_conditions.append(lambda row, c=comparison, v=value: c(row['percent_change_volume_over_last_wk'], v))

    if rule[23] == 1:
        relation = Individual.relational_symbol(rule[24:26])
        value = Individual.bit_to_decimal(rule[26:33])
        comparison = comparison_map[relation]
        rule_conditions.append(lambda row, c=comparison, v=value: c(row['days_to_next_dividend'], v))

    if rule[33] == 1:
        relation = Individual.relational_symbol(rule[34:36])
        value = Individual.bit_to_decimal(rule[36:43]) / 100
        comparison = comparison_map[relation]
        rule_conditions.append(lambda row, c=comparison, v=value: c(row['percent_return_next_dividend'], v))

    def rule_function(row):
        return all(condition(row) for condition in rule_conditions)

    return rule_function
