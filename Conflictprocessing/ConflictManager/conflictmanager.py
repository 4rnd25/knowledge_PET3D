"""
Created on May 07 2025 07:46

@author: ISAC - pettirsch
"""
import os
import pandas as pd

class ConflictManager:

    def __init__(self, mother_output_folder=None, recordIdList = [], recordNameList = [], load_all = True,
                 verbose=False):

        self.mother_output_folder = mother_output_folder
        self.recordIdList = recordIdList
        self.recordNameList = recordNameList
        self.load_all = load_all
        self.verbose = verbose

        self.load_all_conflicts()

    def load_all_conflicts(self):
        """
        Load all conflicts from the output folder.
        """

        df_list = []
        for idx, recordId in enumerate(self.recordIdList):
            record_folder = os.path.join(self.mother_output_folder, str(self.recordNameList[idx]))
            if not os.path.exists(record_folder):
                raise FileNotFoundError(f"Record folder {record_folder} does not exist.")

            record_conflict_folder = os.path.join(record_folder, "Conflicts")
            base = str(self.recordNameList[idx])

            # Try canonical (old) and anonymized (new) spellings
            candidates = [
                os.path.join(record_conflict_folder, f"{base}_Conflicts.csv"),  # old, capital C
                os.path.join(record_conflict_folder, f"{base}_conflicts.csv"),  # new, lower-case c
            ]

            csv_path = next((p for p in candidates if os.path.exists(p)), None)

            # As a last resort, accept any case/extension variant like *_Conflicts.*, *_conflicts.*
            if csv_path is None:
                matches = glob.glob(os.path.join(record_conflict_folder, f"{base}_*onflicts.*"))
                csv_path = matches[0] if matches else None

            if csv_path is None:
                raise FileNotFoundError(
                    f"CSV file not found for base '{base}'. Looked for: "
                    f"{candidates} and glob '{base}_*onflicts.*' in {record_conflict_folder}"
                )

            if self.verbose:
                print(f"Loading conflicts from {csv_path}. Missing recordIDs: {len(self.recordIdList) - idx - 1}")

            df = pd.read_csv(csv_path)
            df_list.append(df)

        # Concatenate all DataFrames into one
        self.df = pd.concat(df_list, ignore_index=True)

        # Ignore entries where "Vehicle_Cluster_1" is -1 or "Vehicle_Cluster_2" is -1
        self.df = self.df[(self.df['Vehicle_Cluster_1'] != -1) & (self.df['Vehicle_Cluster_2'] != -1)]

    def get_all_unique_types(self) -> list:
        """
        Get all unique conflict types from the DataFrame.
        """
        return self.df['Auto_Type'].unique().tolist()

    def get_unique_vehicle_class_combinations(self) -> list:
        """
        Get all unique vehicle class combinations from the DataFrame.
        """
        unique_combinations = self.df[['Vehicle_Class_1', 'Vehicle_Class_2']].drop_duplicates()
        return [tuple(x) for x in unique_combinations.values]

    def get_filtered_conflicts(
            self,
            filterIndicator: str = 'all',
            filterClass1: str = 'all',
            filterClass2: str = 'all',
            filterCluster1: str = 'all',
            filterCluster2: str = 'all',
            Auto_Type: str = 'all',
            auto_rule_flag: str = 'all',
            value: float = 3.0,
            LOF1: float = 0.0,
            LOF2: float = 0.0,
            LOF2_LSTM: float = 0.0,
            LOFOR: bool = False,
            idVehicle1: int = None,
            idVehicle2: int = None
    ) -> pd.DataFrame:
        """
        Return a new DataFrame of conflicts, applying any non-'all' filters.

        - You can pass comma-separated lists, e.g. filterIndicator='PET3D,PET2D'
        - filterCluster1/2 behave like the other categorical filters
        - `value` is used as a numeric threshold: only rows with Value >= value are kept
        """

        # start with all rows included
        mask = pd.Series(True, index=self.df.index)

        # map method args → (column name, raw filter value)
        filter_map = {
            'Indicator': filterIndicator,
            'Vehicle_Class_1': filterClass1,
            'Vehicle_Class_2': filterClass2,
            'Auto_Type': Auto_Type
        }

        # apply all categorical filters (including clusters)
        for col_name, raw_val in filter_map.items():
            if raw_val and raw_val.lower() != 'all':
                wanted = [v.strip().lower() for v in raw_val.split(',') if v.strip()]
                mask &= self.df[col_name].astype(str).str.lower().isin(wanted)

        # apply numeric threshold on the 'Value' column
        if value is not None:
            mask &= self.df['Value'] <= value

        if filterCluster1 != 'all':
            mask &= self.df['Vehicle_Cluster_1'] == float(filterCluster1)

        if filterCluster2 != 'all':
            mask &= self.df['Vehicle_Cluster_2'] == float(filterCluster2)

        if auto_rule_flag != 'all':
            if auto_rule_flag == 'True':
                auto_rule_flag = True
            elif auto_rule_flag == 'False':
                auto_rule_flag = False
            mask &= self.df['Auto_Rule_Flag'] == auto_rule_flag

        if LOFOR:
            mask &= (self.df['LOF1'] > LOF1) | (self.df['LOF2'] > LOF1)
        else:
            if LOF1 > 0:
                mask &= self.df['LOF1'] >= LOF1

            if LOF2 > 0:
                mask &= self.df['LOF2'] >= LOF2

            if LOF2_LSTM > 0:
                mask &= self.df['LOF2_LSTM'] >= LOF2_LSTM

        # apply idVehicle1 and idVehicle2 filters
        if idVehicle1 is not None:
            mask &= self.df['idVehicle1'] == idVehicle1

        if idVehicle2 is not None:
            mask &= self.df['idVehicle2'] == idVehicle2

        # return only the rows that passed all filters
        return self.df.loc[mask].copy()
