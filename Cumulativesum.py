import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
# JUST AN EXAMPLE DATASET FOR PRACTICE
df = pd.DataFrame({'EMG':[0.1,0.2,0.3,0.4,0.5,0.6,0.8,0.93,0.98]})
MVC_value = 1.34 #detect using .max function or something
df.index = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
df['EMG_normalized'] = df['EMG']/MVC_value
# DO NOT PLOT THIS...CUMULATIVE SUM PLOTS BETTER
plt.plot(df['EMG_normalized'], label='EMG')
plt.show()
# cumulative sum of normalized values
df['EMG_cumsum'] = df['EMG_normalized'].cumsum()

# Scale cumulative sum to end at final normalized value
df['EMG_cumsum'] = df['EMG_cumsum']/df['EMG_normalized'].sum() * (df['EMG_normalized'].iloc[-1])

# Scaled cumulative value = (cumsum/normalized values sum) * final normalized percentage
# E.g., 3.59/3.59 * 0.73 = 0.73
plt.plot(df['EMG_cumsum'])
plt.show()


print(df)