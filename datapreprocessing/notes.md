# data analysis

## Spearman method - correlation among parameters

## feature importance
- Adjusted R2 value
Using RandomForestRegressor(100, random_state=42):

5.490447158991832 <- mean_squared_error
0.8851354373341692 <- adj_R2

[('age', 0.3294679334945805),
 ('cement', 0.3208240388388405),
 ('water', 0.1176630918187129),
 ('blast_furnace_slag', 0.07282757696319311),
 ('superplasticizer', 0.07066465816511815),
 ('fine_aggregate ', 0.0382926258469232),
 ('coarse_aggregate', 0.026685268306048344),
 ('fly_ash', 0.0235748065665834)] <- feature importance of each input element

 'coarse_aggregate' and 'fly_ash' seem to not change heavily the output value; reuse of RandomForestRegressor(100, random_state=42) without them.

5.403008276712589 <- mean_squared_error
0.8896512234877525 <- adj_R2

Both values have been improved.

[('cement', 0.3359942014855553),
 ('age', 0.33123455332354856),
 ('water', 0.12247501772358364),
 ('blast_furnace_slag', 0.08650306793059828),
 ('superplasticizer', 0.07980825721229104),
 ('fine_aggregate ', 0.04398490232442321)] <- better feature importances

- Dataset comparison with/without outlayers
Removing outlayers (from 1030 to 930 data samples) we get

5.403008276712589 <- mean_squared_error
0.8896512234877525 <- adj_R2