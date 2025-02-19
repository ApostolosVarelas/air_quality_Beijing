import os
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['pm2.5', 'Iws'])
    return df

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Plot 1: Scatter plots of monthly averages for each year
def plot_monthly_averages(df, output_dir):
    create_directory(output_dir)
    for year in df['year'].unique():
        yearly_data = df[df['year'] == year]
        monthly_avg = yearly_data.groupby('month')[['pm2.5', 'Iws']].mean()

        fig, ax1 = plt.subplots()

        # pm2.5 on primary y-axis
        ax1.plot(monthly_avg.index, monthly_avg['pm2.5'], label='pm2.5', marker='o', color='blue')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('pm2.5', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Iws on secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(monthly_avg.index, monthly_avg['Iws'], label='Iws', marker='o', color='green')
        ax2.set_ylabel('Iws', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        fig.suptitle(f'Monthly Averages for {year}')
        fig.legend(loc="upper right")
        plt.grid()
        plt.savefig(os.path.join(output_dir, f'monthly_averages_{year}.png'))
        plt.close()

# Plot 2: Scatter plots of daily averages for each month in each year
def plot_daily_averages(df, output_dir):
    for year in df['year'].unique():
        yearly_data = df[df['year'] == year]
        for month in yearly_data['month'].unique():
            monthly_data = yearly_data[yearly_data['month'] == month]
            daily_avg = monthly_data.groupby('day')[['pm2.5', 'Iws']].mean()

            month_dir = os.path.join(output_dir, str(year), f'month_{month}')
            create_directory(month_dir)

            fig, ax1 = plt.subplots()

            ax1.plot(daily_avg.index, daily_avg['pm2.5'], label='pm2.5', marker='o', color='blue')
            ax1.set_xlabel('Day')
            ax1.set_ylabel('pm2.5', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            ax2 = ax1.twinx()
            ax2.plot(daily_avg.index, daily_avg['Iws'], label='Iws', marker='o', color='green')
            ax2.set_ylabel('Iws', color='green')
            ax2.tick_params(axis='y', labelcolor='green')

            fig.suptitle(f'Daily Averages for {year}-{month:02d}')
            fig.legend(loc="upper right")
            plt.grid()
            plt.savefig(os.path.join(month_dir, f'daily_averages_{year}_{month:02d}.png'))
            plt.close()

# Plot 3: Hourly plots for each day
def plot_hourly_values(df, output_dir):
    for year in df['year'].unique():
        yearly_data = df[df['year'] == year]
        for month in yearly_data['month'].unique():
            monthly_data = yearly_data[yearly_data['month'] == month]
            for day in monthly_data['day'].unique():
                daily_data = monthly_data[monthly_data['day'] == day]
                hourly_values = daily_data.groupby('hour')[['pm2.5', 'Iws']].mean()

                day_dir = os.path.join(output_dir, str(year), f'month_{month}', f'day_{day}')
                create_directory(day_dir)

                fig, ax1 = plt.subplots()

                ax1.plot(hourly_values.index, hourly_values['pm2.5'], label='pm2.5', marker='o', color='blue')
                ax1.set_xlabel('Hour')
                ax1.set_ylabel('pm2.5', color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')

                ax2 = ax1.twinx()
                ax2.plot(hourly_values.index, hourly_values['Iws'], label='Iws', marker='o', color='green')
                ax2.set_ylabel('Iws', color='green')
                ax2.tick_params(axis='y', labelcolor='green')

                fig.suptitle(f'Hourly Values for {year}-{month:02d}-{day:02d}')
                fig.legend(loc="upper right")
                plt.grid()
                plt.savefig(os.path.join(day_dir, f'hourly_values_{year}_{month:02d}_{day:02d}.png'))
                plt.close()

if __name__ == "__main__":
    input_file = "PRSA_data_2010.1.1-2014.12.31.csv"
    output_dir = "output_plots"

    data = load_data(input_file)

    plot_monthly_averages(data, os.path.join(output_dir, "monthly_averages"))
    plot_daily_averages(data, os.path.join(output_dir, "daily_averages"))
    plot_hourly_values(data, os.path.join(output_dir, "hourly_values"))
