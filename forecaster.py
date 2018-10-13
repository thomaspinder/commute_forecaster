import datapoint
import numpy as np
from tqdm import tqdm
import pandas as pd
import smtplib

"""
This is a simple script to build a forecast of the weather conditions for a commute later in the day. It will not work 
if the hour at which you are commuting in is greater than the hour value of the day's current time. For example, if 
you commute in at 8.30AM and it is now 9AM, the script will not properly function.

For this reason, it is advised to the run the script as a cronjob in the early hours of the morning. To do this:
1. In the terminal type `crontab -e`
2. Append the following to your crontab file `0 1 * * * /path/to/python /path/to/forecaster.py

This will run the forecaster at 1AM every day.
"""


class Forecaster:
    def __init__(self, home_coords, work_coords, api_key, in_time=8, out_time=17):
        self.home_coords = home_coords
        self.work_coords = work_coords
        self.in_time = np.round(in_time, 0).astype(int)
        self.out_time = np.round(out_time, 0).astype(int)
        self.conn = datapoint.connection(api_key=api_key)
        self.testing = True
        if self.testing:
            self.day_num = 1
        else:
            self.day_num = 0
        self._variable_initalise()
        self.results = {}

    def build_results(self):
        in_rain, out_rain= [], []
        in_temp, out_temp = [], []
        in_wind, out_wind = [], []
        in_gust, out_gust = [], []
        for time in tqdm(self.forecast.days[self.day_num].timesteps, desc='Building Forecast'):
            if time.date.hour in self.in_hours:
                in_rain.append(self._rain_prob(time))
                in_temp.append(self._temps(time))
                in_wind.append(self._wind_speed(time))
                in_gust.append(self._gust_speed(time))
            elif time.date.hour in self.out_hours:
                out_rain.append(self._rain_prob(time))
                out_temp.append(self._temps(time))
                out_wind.append(self._wind_speed(time))
                out_gust.append(self._gust_speed(time))
        self.results['in_rain'],  self.results['out_rain'] =  self._weighted_avg(in_rain, out_rain)
        self.results['in_temp'], self.results['out_temp'] = self._weighted_avg(in_temp, out_temp)
        self.results['in_wind'], self.results['out_wind'] = self._weighted_avg(in_wind, out_wind)
        self.results['in_gust'], self.results['out_gust'] = self._weighted_avg(in_gust, out_gust)


    def produce_results(self, print=False, write=False):
        self.build_results()
        if print:
            self._box_print('Commuting in at {}({}), out at {}({})'.format(self._am_or_pm(self.in_time), self.in_direction,
                                                                           self._am_or_pm(self.out_time),
                                                                           self.out_direction))
            self._box_print('Rain In: {}%, Rain out: {}%'.format(np.round(self.results['in_rain'], 1),
                                                                 np.round(self.results['out_rain'], 1)))
            self._box_print('Temperature In: {}degrees, Temperature out: {}degrees'.format(np.round(self.results['in_temp'], 1),
                                                                 np.round(self.results['out_temp'], 1)))
            self._box_print('Wind In: {}mph, Wind out: {}mph'.format(np.round(self.results['in_wind'], 1),
                                                                 np.round(self.results['out_wind'], 1)))
            self._box_print('Gusts In: {}mph, Gusts out: {}mph'.format(np.round(self.results['in_gust'], 1),
                                                                 np.round(self.results['out_gust'], 1)))
        if write:
            self.write_results()

    def _variable_initalise(self):
        self.site = self.conn.get_nearest_site(self.home_coords[0], self.home_coords[1])
        self.forecast = self.conn.get_forecast_for_site(self.site.id, "3hourly")
        self.timesteps = np.array([timestep.date.hour for timestep in self.forecast.days[self.day_num].timesteps])
        self.in_direction = self._get_direction(self.home_coords, self.work_coords)
        self.out_direction = self._get_direction(self.work_coords, self.home_coords)
        self.in_hours = self._either_side(self.in_time, self.timesteps)
        self.out_hours = self._either_side(self.out_time, self.timesteps)

    def write_results(self):
        results_df = pd.DataFrame(list(self.results.items()), columns=['Metric', 'Value'])
        results_df.to_csv('forecast_results.csv', index=False)

    @staticmethod
    def _temps(time_point):
        return time_point.temperature.value

    @staticmethod
    def _wind_speed(time_point):
        return time_point.wind_speed.value

    @staticmethod
    def _gust_speed(time_point):
        return time_point.wind_gust.value

    @staticmethod
    def _rain_prob(time_point):
        return time_point.precipitation.value

    def _weighted_avg(self, in_set, out_set):
        in_avg = ((((self.in_time-self.in_hours[0])*in_set[0]))+((self.in_hours[1]-self.in_time)*in_set[1]))/(self.in_hours[1]-self.in_hours[0])
        out_avg = ((((self.out_time-self.out_hours[0])*out_set[0]))+((self.out_hours[1]-self.out_time)*out_set[1]))/(self.out_hours[1]-self.out_hours[0])
        return in_avg, out_avg

    @staticmethod
    def _box_print(msg):
        """
        Small helper function to print messages to console in a centralised box.
        :param msg: Message to be placed in box
        :type msg: str
        """
        max_len = max(78, len(msg) + 10)
        print('{}'.format('-' * (max_len + 2)))
        print('|{}|'.format(msg.center(max_len)))
        print('{}'.format('-' * (max_len + 2)))

    @staticmethod
    def _am_or_pm(time_value):
        if time_value < 12:
            return '{}AM'.format(time_value)
        else:
            time_value -= 12
            return '{}PM'.format(time_value)

    @staticmethod
    def _either_side(centre_point, steps):
        """Calculate the hour markers either side of given moving hour.

        Parameters
        ----------
        centre_point : int
            Description of parameter `centre_point`.
        steps : NumPy array
            Set of timesteps within a given window of time

        Returns
        -------
        tuple
            Pair of hour markers corresponding to the value immediately left and right of the supplied centre_point value

        """
        right_idx = np.where(steps > centre_point)[0][0]
        left_idx = right_idx - 1
        return (steps[left_idx], steps[right_idx])

    def _get_direction(self, start_coords, end_coords):
        delta_lon = end_coords[0]-start_coords[0]
        delta_lat = end_coords[1]-start_coords[1]
        angle = np.degrees(np.arctan2(delta_lon, delta_lat))
        return self._to_bearing(angle)

    @staticmethod
    def _to_bearing(angle_value):
        orientation = np.abs(angle_value)
        if orientation < 22.5:
            return 'N'
        elif orientation > 337.5 and orientation < 360:
            return 'N'
        elif orientation > 22.6 and orientation < 67.5:
            return 'NNE'
        elif orientation > 67.6 and orientation < 112.5:
            return 'E'
        elif orientation > 112.6 and orientation < 157.5:
            return 'SSE'
        elif orientation > 157.6 and orientation < 202.5:
            return 'S'
        elif orientation > 202.6 and orientation < 247.5:
            return 'SSW'
        elif orientation > 247.6 and orientation < 292.5:
            return 'W'
        elif orientation > 292.6 and orientation < 337.5:
            return 'NNW'

class Sender:
    def __init__(self, recipient_address):
        self.recip = recipient_address

    def initailise_server(self):
        s = smtplib.SMTP(host='smtp.gmail.com', port=465)
        s.login("your username", "your password")
        s.sendmail(
            "t.pinder1994@gmail.com",
            "tompinder@live.co.uk",
            "this message is from python")
        s.quit()

if __name__ == '__main__':
    commuter = Forecaster(home_coords=(-2.788885, 54.039055),work_coords=(-2.784804, 54.008047),
                          api_key="47929aee-38c8-49cc-a476-6a21cc342e4a", in_time=7)
    commuter.produce_results(print=True, write=True)
    # App password - jxbcsqqvddkowimu