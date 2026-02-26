import pandas as pd
import numpy as np


class RewardCalculator():
    def __init__(self, path, alpha=0.25, fast_delivery_alpha=0.75, device_name='cuda'):
        self.alpha = alpha
        self.fast_delivery_alpha = fast_delivery_alpha
        
        store_cus_dp = pd.read_csv(f'{path}/warehouse_dp_pincode_mapping.csv')
        store_cus_dp['percent'] = store_cus_dp['Count']/store_cus_dp.groupby(['WAREHOUSE_CODE','CENTRAL_PINCODE'])['Count'].transform('sum')
        self.store_cus_dp = store_cus_dp.pivot(index=['WAREHOUSE_CODE','CENTRAL_PINCODE'],columns=['DELIVERY_PARTNERS_CODE'],values='percent').replace(np.nan,0)
        
        self.logistics_partners = list(set(store_cus_dp.columns) - set(['WAREHOUSE_CODE','CENTRAL_PINCODE']))
        
        lc_3pl = pd.read_csv(f'{path}/cost_3pl.csv')
        self.lc_3pl_dict = lc_3pl.set_index(['3PL','zone','slab']).to_dict('index')
        self.delivery_partners = lc_3pl['3PL'].unique()
        
        self.vendor_loc2 = pd.read_csv(f'{path}/stores.csv')
        self.vendor_loc2.loc[:, 'actions'] = self.vendor_loc2.sort_values('WAREHOUSE_CODE').reset_index().index
        self.num_warehouses = len(self.vendor_loc2)
        
        self.agg_eta_tat = pd.read_csv(f'{path}/eta_hour.csv')
        self.agg_eta_tat['IS_fast_delivery'] = self.agg_eta_tat.DELIVERY_TYPE.map({'fast_delivery':1,'slow_delivery':0})
        
        self.eta_mean = pd.read_csv(f'{path}/eta_hour.csv')
        self.eta_mean['IS_fast_delivery'] = self.eta_mean.DELIVERY_TYPE.map({'fast_delivery':1,'slow_delivery':0})
        
        self.eta_mean_dict = self.eta_mean.groupby(['WAREHOUSE_CODE','CENTRAL_PINCODE','IS_fast_delivery']).TAT_MEAN.mean().to_dict()
                
        in_rate_cost = pd.read_csv(f'{path}/cost_in.csv')
        in_rate_cost.delivery_type = in_rate_cost.delivery_type.replace(np.nan,'slow_delivery')
        
        self.in_rate_dict = in_rate_cost.sort_values('count',ascending=False).\
            drop_duplicates(['WAREHOUSE_CODE','CENTRAL_PINCODE','delivery_type'],keep='first').\
                set_index(['WAREHOUSE_CODE','CENTRAL_PINCODE','delivery_type']).logistics_cost.to_dict()
        
        self.vendor_city_zone = pd.read_csv(f'{path}/warehouse_city_zone_mapping.csv')
        self.vendor_city_zone = self.vendor_city_zone.sort_values('Count').drop_duplicates(subset=['CENTRAL_PINCODE','WAREHOUSE_CODE'],keep='last')
        
        self.zone_mean = pd.read_csv(f'{path}/zone_mean_eta.csv')

        self.warehouse_pincode_distance = pd.read_csv(f'{path}/pincode_warehouse_distance.csv')

        
    def calculate_cost(self, warehouse, pincode, fast_delivery, zone, dp, paymode, weight, cold_storage):
        if(dp in self.delivery_partners):
            cost = 0
            weight /=1000
            f5w = self.lc_3pl_dict[(dp,zone,0.5)]['cost']
            if weight <= 0.25:
                cost =  self.lc_3pl_dict[(dp,zone,0.25)]['cost']
            elif weight <= 0.5:
                cost = f5w
            else:
                extra_weight = weight - 0.5
                extra_cost = self.lc_3pl_dict[(dp,zone,1)]['cost'] * (extra_weight // 0.5)
                cost =  f5w + extra_cost 
            if(str(paymode).lower() == 'cod'):
                cost+=self.lc_3pl_dict[(dp,'COD',1)]['cost']
            if(cold_storage):
                return cost * 1.5 # can apply correct logistics cost
            return cost
        else:
            if(fast_delivery<=6):
                return self.in_rate_dict.get((warehouse,'fast_delivery',pincode),55)
            if(cold_storage==1):
                return self.in_rate_dict.get((warehouse,'Cold storage',pincode),55)*1.5
            return self.in_rate_dict.get((warehouse,'slow_delivery',pincode),55)
    
    def reward_value(self,nx):
        # normalize tat and cost
        tat_norm = min(1.0, (nx['tat_mean']-1)/(240-1))
        cost_norm = min(1.0, (nx['cost']-18)/(200-18))
        alpha = self.alpha
        k = 2.5
        if(nx['IS_fast_delivery']):
            alpha = self.fast_delivery_alpha
            
        reward = (1-np.e**(-k*tat_norm - 0))*alpha + cost_norm*(1-alpha)
        reward = 1-(reward)
        fast_delivery_penalty = 0
        zero_penalty = 0
        
        P = (np.e**(-0.01*nx['distance_mean'])) if nx['IS_fast_delivery'] else 1
        retail_penalty = 1e-6 if(nx['zone']!='A' and nx['STORE_TYPE']==0 ) else 1
        
        final_reward = reward*100 * P * retail_penalty
        if final_reward < 0:
            print(f"Final_Reward: {final_reward}, Reward: {reward}, P: {P}, tat_norm: {tat_norm}, cost_norm: {cost_norm}, alpha: {alpha}, tat_exp: {np.e**(-2.5*tat_norm)}, KL_DIV: {nx['KL_DIV'][0]}")
            print(nx)
            raise SystemExit(0)
        return final_reward

    def get_reward(self, df):
        ldf = len(df)
        df = df.merge(self.vendor_loc2, on=['actions'], how='left')
        df['WAREHOUSE_CODE']  = df['actions']
        df['year_mon'] = df.ORDER_DATE.map(lambda x: f"{x.date()}"[0:7])
        df['HOUR'] = df['ORDER_DATE'].dt.hour
        
        df = df.merge(
            self.agg_eta_tat[['WAREHOUSE_CODE', 'HOUR', 'CENTRAL_PINCODE', 'IS_fast_delivery', 'TAT_MEAN']],
            on=['WAREHOUSE_CODE', 'HOUR', 'CENTRAL_PINCODE', 'IS_fast_delivery'],
            how='left'
        )
        # Fill missing TAT_MEAN values using eta_mean_dict (vectorized)
        tat_missing = df['TAT_MEAN'].isna()
        
        df.loc[tat_missing, 'TAT_MEAN'] = df.loc[tat_missing, ['WAREHOUSE_CODE', 'CENTRAL_PINCODE', 'IS_fast_delivery']] \
            .apply(lambda x: self.eta_mean_dict.get(tuple(x.values), np.nan), axis=1)
        
        df['tat'] = df['TAT_MEAN'].fillna(180)

        df = df.merge(self.warehouse_pincode_distance,on=['CENTRAL_PINCODE','WAREHOUSE_CODE'],how='left')
        
        df = df.merge(self.vendor_city_zone, on=['WAREHOUSE_CODE', 'CENTRAL_PINCODE'], how='left')
        df.zone = df.zone.replace(np.nan,'D')
        
        # df['zone'] = np.where(df['dp_zone'].isna(), df['zone'], df['dp_zone'])
        
    #     ------------------------ETA---------------------------------------------

    #     ------------------------logistic partner---------------------------------------------
        
        s_c_d_index = self.store_cus_dp.index
        Map = lambda x,y: self.store_cus_dp.loc[x,y].index[np.where(np.random.multinomial(n=1,pvals=self.store_cus_dp.loc[x,y].to_list())==1)[0]][0] if (x, y) in s_c_d_index else 9
        set_logistics_partner = df[['GROUP_ID','WAREHOUSE_CODE','CENTRAL_PINCODE']].drop_duplicates()
        set_logistics_partner['DELIVERY_PARTNERS_CODE_D'] = set_logistics_partner.apply(lambda x: Map(x['WAREHOUSE_CODE'], x['CENTRAL_PINCODE']),axis=1)      
        df = df.merge(set_logistics_partner,on=['GROUP_ID','WAREHOUSE_CODE','CENTRAL_PINCODE'],how='left')      
        
    
        # Grouping and aggregation
        group_cols = [
            'GROUP_ID', 'WAREHOUSE_CODE', 'CENTRAL_PINCODE', 'IS_fast_delivery', 'tat',
            'zone', 'DELIVERY_PARTNERS_CODE_D', 'PAYMENT_METHOD', 'COLD_STORAGE'
        ]
        sml3 = df.groupby(group_cols, dropna=False).agg({
            'ITEM_WEIGHT_GMS': 'sum',
            'SKU_ID': 'count'
        }).reset_index()

         # Calculate cost
        
        sml3['cost'] = sml3.apply(lambda x:self.calculate_cost(x['WAREHOUSE_CODE'],x['CENTRAL_PINCODE'],x['tat'],x['zone'],x['DELIVERY_PARTNERS_CODE_D'],x['PAYMENT_METHOD'],x['ITEM_WEIGHT_GMS'],x['COLD_STORAGE']),axis=1)

        sml3['cost_level'] = sml3['cost'] / sml3['SKU_ID']

        # Merge cost and other metrics back to the main DataFrame
        df = df.merge(
            sml3[['GROUP_ID', 'WAREHOUSE_CODE', 'CENTRAL_PINCODE', 'IS_fast_delivery', 'cost', 'tat', 'cost_level', 'COLD_STORAGE']],
            on=['GROUP_ID', 'CENTRAL_PINCODE', 'IS_fast_delivery', 'tat', 'WAREHOUSE_CODE', 'COLD_STORAGE'],
            how='left',
            suffixes=('', '_x')
        )
        # Compute cost, tat_mean, and distance_mean (vectorized)
        df['cost'] = df.groupby('GROUP_ID')['cost_level'].transform('sum')
        df['tat_mean'] = df.groupby('GROUP_ID')['tat'].transform('mean')
        df['distance_mean'] = df.groupby('GROUP_ID')['distance'].transform('mean')

        # Compute IS_FULFILLED (vectorized)
        # df['IS_FULFILLED'] = df.apply(lambda x: 0 if x['inputs'][-(self.num_warehouses-x['actions'])] else 1,axis=1)

        # Calculate rewards 
    
        df['reward'] = df.apply(self.reward_value, axis=1)
        return df
    
