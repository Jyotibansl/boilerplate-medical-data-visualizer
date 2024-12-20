import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
medicaldata_path = 'medical_examination.csv'
df = pd.read_csv(medicaldata_path)

# 2
df['overweight'] = (df['weight']/(df['height']/100) **2).apply(lambda x:1 if x>25 else 0)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x:0 if x==1 else 1)
df['gluc'] = df['gluc'].apply(lambda x:0 if x==1 else 1)
df.to_csv('normalized_data_file.csv', index=False)
print(df)

# 4
def draw_cat_plot():

    # 5
    # Prepare the data 
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])


    # 6
    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. 
    df_cat['total'] = 1 
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).count() 
    

    # 7
    # Draw the catplot with 'sns.catplot()' 
    cat_plot = sns.catplot( x="variable", y="total", hue="value", col="cardio", data=df_cat, kind="bar", height=5, aspect=1 )    
    # Customize the plot 
    cat_plot.set_axis_labels("variable", "total") 
    cat_plot.set_titles("cardio {col_name}") 

    # 8
    # Save the plot 
    fig = cat_plot.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
    (df['height'] >= df['height'].quantile(0.025)) &
    (df['height'] <= df['height'].quantile(0.975)) &
    (df['weight'] >= df['weight'].quantile(0.025)) &
    (df['weight'] <= df['weight'].quantile(0.975)) ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(corr)



    # 14
    fig, ax = plt.subplots(figsize=(10, 8))

    # 15
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", vmin=-0.15, vmax=0.3, linewidths=.5)


    # 16
    fig.savefig('heatmap.png')
    return fig
