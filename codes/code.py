df = pd.read_csv(r'C:\Users\91900\Downloads\EDA_Body Cleansing_sample.csv')

df.columns = ['UL_GEO_ID', 'CATG_CD', 'SUBCAT_CD', 'Features', 'Independents',
       'Dependents', 'Constraint', '13_cor', '13_ptop', '13_vif', 'Columns_to_Keep']

def config_read(data = df, cat=3, subcat=13, target_col='PRICE_PER_VOL'):
    '''
    This function aims to read input file from business experts and extract columns and constraint from them.
    The idea to use this function is to automate reading of business input and results in columns and constraint
    for price, tdp and sales.
    
    Columns to be used in input file:
    'UL_GEO_ID',
    'CATG_CD', 
    'SUBCAT_CD', 
    'Independents',
    'Dependents',
    'Constraint'
    'Columns_to_Keep'
    
    '''
    feat = df[(df['Independents']==target_col)&(df['CATG_CD']==cat)&(df['SUBCAT_CD']==subcat)&(df['Columns_to_Keep']==1)]['Dependents'].to_list()
    cons = tuple(df[(df['Independents']==target_col)&(df['CATG_CD']==cat)&(df['SUBCAT_CD']==subcat)&(df['Columns_to_Keep']==1)]['Constraint'].to_list())
    return feat, str(cons)


feat = []
cons = []
for cat in df.CATG_CD.unique():
    for subcat in df.SUBCAT_CD.unique():
        for i in df.Independents.unique(): 
            mx = config_read(df, cat, subcat, i)
            feat.append(mx[0])
            
            cons.append(mx[1])
            
            
        price_input_col = feat[0]
        tdp_input_col = feat[2]
        sales_input_col = feat[1]
        price_cons = cons[0]
        tdp_cons = cons[2]
        sales_cons = cons[1]    
        
        
        
        
[20:09] Suyash Mishra
feat = []
cons = []
out = []
for cat in df.CATG_CD.unique():
for subcat in df.SUBCAT_CD.unique():
for i in df.Independents.unique():
mx = config_read(df, cat, subcat, i)
feat.append(mx[0])

cons.append(mx[1])
out.append(i)
price_input_col = feat[0]
tdp_input_col = feat[2]
sales_input_col = feat[1]
price_cons = cons[0]
tdp_cons = cons[2]
sales_cons = cons[1]
price_output_col = out[0]
tdp_output_col = out[2]
sales_output_col = out[1]\

