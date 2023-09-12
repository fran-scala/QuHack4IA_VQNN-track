# data analysis
Benchmark: random forest regressor in un computer classico
0.135 <- mean_squared_error
0.888 <- adj_R2

## univariata

## multivariata (Spearman method - correlation among parameters)

## studio/rimozione degli outliers (~10%)
Rimozione degli outliers sulle features che possiedono outliers e che non si distribuiscono secondo una Gaussiana 
(q1, q3) = (25 percentile, 75 percentile), sia iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
Removing outliers (no normalization) >= lower_bound and <= upper_bound, we reduce from 1030 to 930 data samples
4.967826664981275 <- mean_squared_error
0.9068670660806136 <- adj_R2

Removing outliers (with normalization) >= lower_bound and <= upper_bound, we reduce from 1030 to 930 data samples
0.12581881198778974 <- mean_squared_error
0.9057977338317024 <- adj_R2

## normalization
Utilizzo di un minimax scaler con min=-1 e max=1 (valori dettati dall'uso della sfera di Bloch)
0.13650391316478677 <- mean_squared_error
0.8856312515340603 <- adj_R2

## feature importance
- Adjusted R2 value
Using RandomForestRegressor(100, random_state=42) we get

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

 'fly_ash' non va a cambiare l'output (rete neurale/PCA stessi risultati con/senza 'fly_ash')
 
