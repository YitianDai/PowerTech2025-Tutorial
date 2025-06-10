# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 10:44:32 2024

The code in WindStorm.py provides the methods required to model how windstorms
can impact a power system

@author: Eduardo Alejandro Martinez Cesena
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
import math
import random
import numpy as np


class Object(object):
    pass


class WindConfig:
    '''Default settings used for the Wind class'''

    def __init__(self):
        # Default time-step and map
        self.data = Object()
        self.data.seed = None

        # Monte Carlo simulations
        self.data.MC = Object()

        self.data.MC.trials = 1000  # Number of trials

        # WindStorms
        self.data.W = Object()

        self.data.num_hrs_yr = 8760  # Hours in a year

        # WindStorms@
        self.data.W.event = Object()
        self.data.W.event.max_yr = 3  # Maximum number of events per year
        self.data.W.event.max_v = [20, 55]  # Min/Max peak wind gust speed
        self.data.W.event.lng = [4, 48]  # Min/Max duration of windstorm
        self.data.W.event.ttr = [24, 168]  # Min/Max restoration time

        # WindStorms: Contour points
        self.data.W.contour = Object()
        self.data.W.contour.lon = [-2.0, -3.3, -3.3, -4.8, -4.8, -3.2, -2.2,
                                   -5.4, -3.2, -5.4, -5.4, 0.4]  # Longitude
        self.data.W.contour.lat = [55.8, 55.0, 53.5, 53.5, 52.8, 52.8, 52.1,
                                   51.8, 51.3, 50.6, 49.9, 50.4]  # Latitude
        self.data.W.contour.from_to = \
            [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
             [9, 10], [10, 11], [11, 12]]  # Connectivity
        self.data.W.contour.dlon = [-1.5, 2.1]  # End points (longitude)
        self.data.W.contour.dlat = [-17/18, 54.183333]  # End points (latitude)

        # Fragility curve settings
        self.data.frg = Object()
        self.data.frg.mu = 3.8
        self.data.frg.sigma = 0.122
        self.data.frg.thrd_1 = 20
        self.data.frg.thrd_2 = 90
        self.data.frg.shift_f = 0

    def set_seed(self, val):
        '''Set seed'''
        self.data.seed = val


class WindClass:
    def __init__(self, obj=None):
        '''Initialise WindStorm class'''

        # Get default values
        if obj is None:
            obj = WindConfig()

        # Copy attributes
        for pars in obj.__dict__.keys():
            setattr(self, pars, getattr(obj, pars))

        # Gets
        lim_max_v_ws = self._get_lim_max_v_ws()
        max_ws_yr = self._get_max_ws_yr()
        lim_lng_ws = self._get_lim_lng_ws()

        # Setting seed
        random.seed(a=self.data.seed, version=2)

        # Settings for MC simulation
        self.MC = Object()

        # Description of windstorm
        self.MC.W = Object()
        self.MC.W.num_yr = \
            [random.randint(1, max_ws_yr) for i in
             range(self.data.MC.trials)]  # Events per year
        self.MC.W.total = sum(self.MC.W.num_yr)  # number of events
        self.MC.W.v = \
            [[lim_max_v_ws[0] + random.random() *
              (lim_max_v_ws[1]-lim_max_v_ws[0])
              for i in range(self.MC.W.total)],
             [17 for i in range(self.MC.W.total)]]  # Max/Min speeds
        self.MC.W.lng = \
            [random.randint(lim_lng_ws[0], lim_lng_ws[1])
             for i in range(self.MC.W.total)]  # Duration of wind storm

    def _crt_bgn_hr(self):
        '''Define hour when wind storm begins per year'''

        # Gets
        lim_lng_ws = self._get_lim_lng_ws()
        lim_ttr = self._get_lim_ttr()
        max_ws_yr = self._get_max_ws_yr()
        num_ws_yr = self._get_num_ws_yr()
        num_hrs_yr = self._get_num_hrs_yr()

        max_lng = max(lim_lng_ws) + max(lim_ttr)
        bgn_hr_ws_yr = np.zeros((len(num_ws_yr), max_ws_yr))
        # bgn_hr_ws_yr = [[0 for i in range(num_ws_total)]
        #                 for j in range(max_ws_yr)]
        for i in range(len(num_ws_yr)):
            for j in range(num_ws_yr[i]):
                # Find inital point in the range
                if j == 0:
                    rn1 = 1
                else:
                    rn1 = bgn_hr_ws_yr[i][j-1] + max_lng
                # Find final point in the range
                rn2 = num_hrs_yr - (num_ws_yr[i] - j+1)*max_lng
                bgn_hr_ws_yr[i][j] = random.randint(int(rn1), int(rn2))

        # Sets
        self._set_bgn_hr_ws_yr(bgn_hr_ws_yr)

    def _get_bgn_hr_ws_yr(self):
        '''Get bgn_hr_ws_yr'''
        return self.MC.W.hrs_yr

    def _get_cp_lat_n(self):
        '''Get cp_lat_n'''
        return self.data.W.contour.dlat

    def _get_cp_lon_n(self):
        '''Get cp_lon_n'''
        return self.data.W.contour.dlon

    def _get_lim_lng_ws(self):
        '''Get lim_lng_ws'''
        return self.data.W.event.lng

    def _get_lim_max_v_ws(self):
        '''Get lim_max_v_ws'''
        return self.data.W.event.max_v

    def _get_lim_ttr(self):
        '''Get lim_ttr'''
        return self.data.W.event.ttr

    def _get_lng_ws(self):
        '''Get lng_ws'''
        return self.MC.W.lng

    def _get_cp_dis_aggregated(self):
        '''Get cp_dis_aggregated'''
        return self.data.W.contour.dis

    def _get_cp_from_to(self):
        '''Get cp_from_to'''
        return self.data.W.contour.from_to

    def _get_cp_lat(self):
        '''Get cp_lat'''
        return self.data.W.contour.lat

    def _get_cp_lon(self):
        '''Get cp_lon'''
        return self.data.W.contour.lon

    def _get_cp_num(self):
        '''Get cp_num'''
        return self.data.W.contour.num

    def _get_lim_v_ws(self):
        '''Get lim_v_ws'''
        return self.MC.W.v

    def _get_max_ws_yr(self):
        '''Get max_ws_yr'''
        return self.data.W.event.max_yr

    def _get_num_hrs_yr(self):
        '''Get num_hrs_yr'''
        return self.data.num_hrs_yr

    def _get_num_ws_total(self):
        '''Get num_ws_total'''
        return self.MC.W.total

    def _get_num_ws_yr(self):
        '''Get num_ws_yr'''
        return self.MC.W.num_yr
    
    def _get_mcs_yr(self):
        '''Get num_mcs_yr'''
        return self.data.MC.trials
    
    def _get_lng_ws(self):
        '''Get lng_ws'''
        return self.MC.W.lng

    def _get_frg_mu(self):
        '''Get frg_mu'''
        return self.data.frg.mu
    
    def _get_frg_sigma(self):
        '''Get frg_sigma'''
        return self.data.frg.sigma
    
    def _get_frg_thrd_1(self):
        '''Get frg_thrd_1'''
        return self.data.frg.thrd_1
    
    def _get_frg_thrd_2(self):
        '''Get frg_thrd_2'''
        return self.data.frg.thrd_2
    
    def _get_frg_shift_f(self):
        '''Get frg_shift_f'''
        return self.data.frg.shift_f

    def _getDistance(self, Lon1, Lat1, Lon2, Lat2):
        '''Get distance between two coordinates [km]'''
        R = 6371000  # Earth Radious [m]
        L1 = Lat1 * math.pi/180  # Radians
        L2 = Lat2 * math.pi/180  # Radians
        DL = (Lat2-Lat1) * math.pi/180
        DN = (Lon2-Lon1) * math.pi/180

        a = math.sin(DL/2) * math.sin(DL/2) + math.cos(L1) * math.cos(L2) * \
            math.sin(DN/2) * math.sin(DN/2)
        c = 2 * math.atan(math.sqrt(a)/math.sqrt(1-a))

        d = R * c/1000  # [km]

        return d
    
    def _getBearing(self, Lon1, Lat1, Lon2, Lat2):
        '''Get the geographical bearing between two coordinates in radian'''
        phi_1 = Lat1 * math.pi/180  # Radians
        phi_2 = Lat2 * math.pi/180  # Radians
        lambda_1 = Lon1 * math.pi/180 # Radians
        lambda_2 = Lon2 * math.pi/180 # Radians

        delta_lambda = lambda_2-lambda_1
        y = math.sin(delta_lambda) * math.cos(phi_2)
        x = math.cos(phi_1) * math.sin(phi_2) - math.sin(phi_1) * \
            math.cos(phi_2) * math.cos(delta_lambda)

        theta = math.atan2(y, x)

        alpha = (theta + 2 * math.pi) % (2 * math.pi)

        return alpha # Radians

    def _getDestination(self, Lon1, Lat1, bearing, distance):
        '''Get the destination's coordinates based on the starting point's coordinates, 
        the travelling direction and distance [m]'''
        max_trials = 10
        tolerance = 1e-3

        # The ubiquitous WGS-84 is a geocentric datum, based on an ellipsoid with:
        a = 6378137.0  # metres - Semi-major axis
        b = 6356752.314245  # metres - Semi-minor axis
        f = (a - b) / a  # Flattening

        phi_1 = Lat1 * math.pi/180 # Radians
        lambda_1 = Lon1 * math.pi/180 # Radians
        alpha_1 = bearing

        s_alpha_1 = math.sin(alpha_1)
        c_alpha_1 = math.cos(alpha_1)
        t_u1 = (1 - f) * math.tan(phi_1)
        c_u1 = 1 / math.sqrt(1 + t_u1 * t_u1)
        s_u1 = t_u1 * c_u1
        sigma_1 = math.atan2(t_u1, c_alpha_1)
        s_alpha = c_u1 * s_alpha_1
        c_sq_alpha = 1 - s_alpha * s_alpha
        u_sq = c_sq_alpha * (a * a - b * b) / (b * b)
        A = 1 + (u_sq / 16384) * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
        B = (u_sq / 1024) * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))

        sigma = distance / (b * A)
        sigma_p = 0
        for _ in range(max_trials):
            c_2_sigma_m = math.cos(2 * sigma_1 + sigma)
            s_sigma = math.sin(sigma)
            c_sigma = math.cos(sigma)
            delta_sigma = B * s_sigma * (c_2_sigma_m + (B / 4) * (c_sigma * \
                        (-1 + 2 * c_2_sigma_m * c_2_sigma_m) - (B / 6) * c_2_sigma_m * \
                            (-3 + 4 * s_sigma * s_sigma) * (-3 + 4 * c_2_sigma_m * c_2_sigma_m)))
            sigma_p = sigma
            sigma = distance / (b * A) + delta_sigma
            if abs(sigma - sigma_p) < tolerance:
                break

        xx = s_u1 * s_sigma - c_u1 * c_sigma * c_alpha_1
        phi_d = math.atan2(s_u1 * c_sigma + c_u1 * s_sigma * c_alpha_1,
                            (1 - f) * math.sqrt(s_alpha * s_alpha + xx * xx))
        lambda_ = math.atan2(s_sigma * s_alpha_1, c_u1 * c_sigma - s_u1 * \
                             s_sigma * c_alpha_1)
        C = (f / 16) * c_sq_alpha * (4 + f * (4 - 3 * c_sq_alpha))
        L = lambda_ - (1 - C) * f * s_alpha * (sigma + C * s_sigma * \
            (c_2_sigma_m + C * c_sigma * (-1 + 2 * c_2_sigma_m * c_2_sigma_m)))
        lambda_d = lambda_1 + L

        d_lon_lat = [math.degrees(lambda_d), math.degrees(phi_d)]
        if _ == max_trials - 1:
            print('Warning: Not meeting tolerance')

        return d_lon_lat

    def _init_ws_path0(self):
        '''Preparations to define starting point of wind storm'''

        # Gets
        cp_from_to = self._get_cp_from_to()
        cp_lat = self._get_cp_lat()
        cp_lon = self._get_cp_lon()

        # Getting distances for each segment of the contour
        cp_num = len(cp_from_to)
        dis = [0 for i in range(cp_num+1)]
        
        for i in range(cp_num):
            f = cp_from_to[i][0]-1
            t = cp_from_to[i][1]-1
            d = self._getDistance(cp_lon[f], cp_lat[f], cp_lon[t], cp_lat[t])
            dis[i+1] = dis[i] + d
        self.data.W.contour.num = cp_num
        self.data.W.contour.dis = [dis[i]/dis[cp_num] for i in range(cp_num+1)]

    def _init_ws_path(self, NumWS):
        '''Defining starting point and direction of wind storm'''

        # Gets
        cp_lat = self._get_cp_lat()
        cp_lon = self._get_cp_lon()
        cp_lon_n = self._get_cp_lon_n()
        cp_lat_n = self._get_cp_lat_n()
        cp_dis_aggregated = self._get_cp_dis_aggregated()
        Rnd_Location = [random.random() for i in range(NumWS)]
        Rnd_direction = [random.random() for i in range(NumWS)]

        # Random starting point
        Lon = [0 for i in range(NumWS)]
        Lat = [0 for i in range(NumWS)]
        for i in range(NumWS):
            j = 1
            while Rnd_Location[i] > cp_dis_aggregated[j]:
                j += 1

            aux = (Rnd_Location[i] - cp_dis_aggregated[j-1]) / \
                (cp_dis_aggregated[j] - cp_dis_aggregated[j-1])
            Lon[i] = cp_lon[j-1] + aux * (cp_lon[j] - cp_lon[j-1])
            Lat[i] = cp_lat[j-1] + aux * (cp_lat[j] - cp_lat[j-1])

        # Random direction
        Lon_n = [0 for i in range(NumWS)]
        Lat_n = [0 for i in range(NumWS)]
        for i in range(NumWS):
            Lon_n[i] = cp_lon_n[1] - \
                Rnd_direction[i]*(cp_lon_n[1] - cp_lon_n[0])
            Lat_n[i] = Lon_n[i]*cp_lat_n[0] + cp_lat_n[1]

        return Lon, Lat, Lon_n, Lat_n
    
    def linear_interpolate(self, start, end, num_points):
        ''' Linearly interpolate between start and end '''
        return [start + i * (end - start) / (num_points - 1) for i in range(num_points)]

    def cubic_interpolate(self, Lon1, Lat1, Lon2, Lat2):
        from scipy.interpolate import CubicSpline

        reverse = Lon1 > Lon2
        # Ensure x is always in increasing order for interpolation
        if reverse:
            Lon1, Lon2 = Lon2, Lon1
            Lat1, Lat2 = Lat2, Lat1
        
        cs = CubicSpline([Lon1, Lon2], [Lat1, Lat2], bc_type=((1, 0), (1, 0)))
        return cs

    def _crt_ws_path(self, Lon1, Lat1, Lon2, Lat2, Num_hrs):
        '''Create the propagation path of a windstorm on an hourly basis'''
        # trajectory of windstorm
        dir_lon = self.linear_interpolate(Lon1, Lon2, Num_hrs + 1)
        cs = self.cubic_interpolate(Lon1, Lat1, Lon2, Lat2)
        dir_lat = cs(dir_lon)

        path_ws = [[0, 0] for _ in range(Num_hrs + 1)]
        path_ws[0] = [Lon1, Lat1]

        for hr in range(1, Num_hrs + 1):
            dist_hr = 24000 - 8000 * (hr - 1) / Num_hrs
            brg_hr = self._getBearing(dir_lon[hr - 1], dir_lat[hr - 1], dir_lon[hr], dir_lat[hr])
            path_ws[hr] = self._getDestination(path_ws[hr - 1][0], path_ws[hr - 1][1], brg_hr, dist_hr)
        
        return path_ws

    def _crt_ws_v(self, lim_v_ws, Num_hrs):
        '''wind gust speeds of a wind storm at each hour'''
        a = math.log(lim_v_ws[1] / lim_v_ws[0]) / (Num_hrs - 1)

        v_ws = [lim_v_ws[0] * math.exp(a * i) for i in range(Num_hrs)]

        return v_ws

    def _compare_circle(self, epicentre, rad_ws, gis_bgn, gis_end, Num_bch):
        '''identify whether an asset falls within the impact zone 
        marked by a radius [km] around the epicentre'''

        Flgs = [False] * Num_bch
        for xt in range(Num_bch):
            if gis_bgn[xt][0] == gis_end[xt][0]:  # Special case, vertical line
                x = gis_bgn[xt][0]
                aux = max(gis_bgn[xt][1], gis_end[xt][1])
                if epicentre[1] > aux:
                    y = aux
                else:
                    aux = min(gis_bgn[xt][1], gis_end[xt][1])
                    if epicentre[1] < aux:
                        y = aux
                    else:
                        y = epicentre[1]
            else:
                b = (gis_bgn[xt][1] - gis_end[xt][1]) / (gis_bgn[xt][0] - gis_end[xt][0])
                a = gis_bgn[xt][1] - gis_bgn[xt][0] * b
                x = (b * (epicentre[1] - a) + epicentre[0]) / (b**2 + 1)
                aux = max(gis_bgn[xt][0], gis_end[xt][0])
                if x > aux:
                    x = aux
                else:
                    aux = min(gis_bgn[xt][0], gis_end[xt][0])
                    if x < aux:
                        x = aux
                y = a + b * x
            # Calculate distance and check if within radius
            if self._getDistance(epicentre[0],epicentre[1], x, y) < rad_ws:
                Flgs[xt] = True
        return Flgs

    def _crt_envelope(self, epicentre, epicentre1, rad_ws):
        '''Create an envelope of coordinates around a path defined by two points'''
        evlp_pts = []
        
        # Calculate initial bearing from the epicentre to epicentre1
        alpha_0 = self._getBearing(epicentre[0], epicentre[1], epicentre1[0], epicentre1[1])
        
        # Calculate envelope points
        rad_ws = rad_ws * 1000 # change to [meters]
        evlp_pts.append(self._getDestination(epicentre[0], epicentre[1], alpha_0 - 90* math.pi/180, rad_ws))
        evlp_pts.append(self._getDestination(epicentre[0], epicentre[1], alpha_0 + 90* math.pi/180, rad_ws))
        evlp_pts.append(self._getDestination(epicentre1[0], epicentre1[1], alpha_0 - 90* math.pi/180, rad_ws))
        evlp_pts.append(self._getDestination(epicentre1[0], epicentre1[1], alpha_0 + 90* math.pi/180, rad_ws))
        
        return evlp_pts

    def _compare_envelope(self, evlp, gis_bgn, gis_end, Num_bch):
        '''Identify groups of lines that are within the envelope'''

        # Assumed sequence
        evlp_sequence = [0, 2, 3, 1, 0]
        evlp = np.array(evlp)

        x = evlp[evlp_sequence, 0]
        y = evlp[evlp_sequence, 1]

        # Get line equations and angle between lines
        aux = np.array([[1, 0], [2, 1], [3, 2], [4, 3]])
        for i in np.where(x[1:] == x[:-1])[0]:
            x[aux[i, 0]] -= 0.000001
            x[aux[i, 1]] += 0.000001
        
        for i in np.where(y[1:] == x[:-1])[0]:
            y[aux[i, 0]] -= 0.000001
            y[aux[i, 1]] += 0.000001
        
        b = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        a = y[:-1] - b * x[:-1]

        # XY range of each line
        aux1 = np.column_stack((x, np.roll(x, -1)))
        aux2 = np.column_stack((y, np.roll(y, -1)))
        xy_range = np.column_stack((np.min(aux1, axis=1), np.max(aux1, axis=1), np.min(aux2, axis=1), np.max(aux2, axis=1)))
        
        flgs = np.zeros(Num_bch, dtype=bool)
        for i in range(Num_bch):
            flgs[i] = self._compare_envelope_line(xy_range, a, b, gis_bgn[i], gis_end[i])
        
        return flgs
    
    def _compare_envelope_line(self, xy_range, a, b, bgn, end):
        '''Auxiliary function to identify if a line is within an envelope'''

        bgn = list(bgn)
        end = list(end)

        # Adjusting values to avoid division by zero
        if end[0]==bgn[0]:
            end[0] = end[0] +0.00001
            bgn[0] = bgn[0] -0.00001

        d = (end[1]-bgn[1]) / (end[0]-bgn[0])
        c = end[1] - d * end[0]
        
        # Intersection points
        x = (a - c) / (d - b)
        y = a + b * x
        
        flg = np.zeros((6, 4))
        xy_line_range = [np.min([bgn[0], end[0]]),  # Min longitude
                        np.max([bgn[0], end[0]]),  # Max longitude
                        np.min([bgn[1], end[1]]),  # Min latitude
                        np.max([bgn[1], end[1]])]   # Max latitude
        
        for i in range(4):
            if xy_range[i, 0] <= x[i] <= xy_range[i, 1]:
                flg[2, i] = 1
                if xy_line_range[0] <= x[i] <= xy_line_range[1]:
                    flg[0, i] = 1
                elif x[i] >= xy_line_range[1]:
                    flg[4, i] = 1
                elif x[i] <= xy_line_range[0]:
                    flg[4, i] = -1
                    
            if xy_range[i, 2] <= y[i] <= xy_range[i, 3]:
                flg[3, i] = 1
                if xy_line_range[2] <= y[i] <= xy_line_range[3]:
                    flg[1, i] = 1
                elif y[i] >= xy_line_range[3]:
                    flg[5, i] = 1
                elif y[i] <= xy_line_range[2]:
                    flg[5, i] = -1
        
            in_out = False
            if np.sum(flg[0:2, :]) > 0:
                in_out = True
            elif np.sum(flg[2, :]) > 0:
                if np.min(flg[4, flg[2, :] == 1]) == -1 and np.max(flg[4, flg[2, :] == 1]) == 1:
                    in_out = True
            elif np.sum(flg[4, :]) > 0:
                if np.min(flg[5, flg[3, :] == 1]) == -1 and np.max(flg[5, flg[3, :] == 1]) == 1:
                    in_out = True
                    
            return in_out

    def _fragility_curve(self, hzd_int):
        '''Calculate the probability of failure based on a fragility curve'''
        
        from scipy.stats import lognorm

         # Gets
        mu = self._get_frg_mu()
        sigma = self._get_frg_sigma()
        thrd_1 = self._get_frg_thrd_1()
        thrd_2 = self._get_frg_thrd_2()
        shift_f = self._get_frg_shift_f()

        f_hzd_int = hzd_int - shift_f
    
        if f_hzd_int < thrd_1:
            pof = 0
        elif f_hzd_int > thrd_2:
            pof = 1
        else:
            # Convert mu and sigma for lognormal distribution
            shape = sigma
            scale = np.exp(mu)
            # Calculate the cumulative distribution function (CDF) of the lognormal distribution
            pof = lognorm.cdf(f_hzd_int, s=shape, scale=scale)
        
        return pof

    def _set_bgn_hr_ws_yr(self, val):
        '''Set cp_lat_n'''
        self.MC.W.hrs_yr = val
    
    def _ws_evlp_plot(self, net, evlp_total, path_ws):
        import matplotlib.pyplot as plt
        import pandapower.plotting as plot
        fig, ax = plt.subplots()
        bc = plot.create_bus_collection(net, buses=net.bus.index, color='black', size=0.02, zorder=1)
        lc = plot.create_line_collection(net, lines=net.line.index,alpha=0.5, color='grey', zorder=2)
        collections_to_draw = [lc, bc]

        for i, evlp in enumerate(evlp_total):
            ax.plot(path_ws[i][0], path_ws[i][1], 'o',alpha=0.5, color='tab:blue', markersize=2)
            evlp_sequence = [0, 2, 3, 1, 0]
            evlp = np.array(evlp)
            x = evlp[evlp_sequence, 0]
            y = evlp[evlp_sequence, 1]
            ax.plot(x,y, '-',alpha=0.5, color = 'tab:blue', linewidth=1, label='Path')

        ax.set_aspect('equal')
        plot.draw_collections(collections_to_draw, ax=ax)
        plt.show()

        fig, ax = plt.subplots()
        bc = plot.create_bus_collection(net, buses=net.bus.index, color='black', size=0.02, zorder=1)
        lc = plot.create_line_collection(net, lines=net.line.index,alpha=0.5, color='grey', zorder=2)
        collections_to_draw = [lc, bc]

        for i, evlp in enumerate(evlp_total):
            ax.plot(path_ws[i][0], path_ws[i][1], 'o',alpha=0.5, color='tab:blue',markerfacecolor='None', markersize=20)

        ax.set_aspect('equal')
        plot.draw_collections(collections_to_draw, ax=ax)
        plt.show()
    
    def _ws_animation(self, net, evlp_total, affected_total, failed_total, Num_hrs, path_ws):

        import matplotlib.pyplot as plt
        import pandapower.plotting as plot
        import imageio.v2 as imageio
        import os

        plt.ioff() 
        evlp_items = [item for item in evlp_total]
        affected_items = [item for item in affected_total]
        failed_items = [item for item in failed_total]
        x, y, x1, y1, total_failed = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        # Create static collections for buses and lines
        bc = plot.create_bus_collection(net, buses=net.bus.index, color='black', size=0.02, zorder=1)
        lc = plot.create_line_collection(net, lines=net.line.index,alpha=0.5, color='grey', zorder=2)
        collections_to_draw = [lc, bc]

        for i in range(Num_hrs):
            collections_to_draw_now = collections_to_draw.copy()
            fig, ax = plt.subplots()

            # Update and plot the path coordinates up to the current hour
            x1 = np.append(x1, path_ws[i][0])
            y1 = np.append(y1, path_ws[i][1])
            ax.plot(x1, y1, 'o',alpha=0.5, color='tab:blue',markerfacecolor='None', markersize=1)
            
            # Process affected and failed lines for the current hour
            current_evlp_array = np.array(evlp_items[i])
            current_affected_array = np.array(affected_items[i])
            current_failed_array = np.array(failed_items[i])
            affected_only_array = [item for item in current_affected_array if item not in total_failed]
            
            # Plot the envelope for the current hour
            evlp_sequence = [0, 2, 3, 1, 0]
            x = np.append(x, current_evlp_array[evlp_sequence, 0])
            y = np.append(y, current_evlp_array[evlp_sequence, 1])
            ax.plot(x,y, '-',alpha=0.5, color = 'tab:blue', linewidth=1, label='Path')

            # Create collections for affected and failed lines if they exist
            if current_affected_array.size > 0 :
                if current_failed_array.size > 0 :
                    total_failed = np.append(total_failed, current_failed_array) 
                    lc1 = plot.create_line_collection(net, lines=affected_only_array, color='#FFCC99')
                    lc2 = plot.create_line_collection(net, lines=total_failed, color='red')
                    collections_to_draw.append(lc1)
                    collections_to_draw.append(lc2)
                    collections_to_draw_now = collections_to_draw.copy()
                    lc1_now = plot.create_line_collection(net, lines=affected_only_array, color='orange')
                    lc2_now = plot.create_line_collection(net, lines=total_failed, color='red')
                    collections_to_draw_now.append(lc1_now)
                    collections_to_draw_now.append(lc2_now)
                else:
                    lc1 = plot.create_line_collection(net, lines=affected_only_array, color='#FFCC99')
                    lc2 = plot.create_line_collection(net, lines=total_failed, color='red')
                    collections_to_draw.append(lc1)
                    collections_to_draw.append(lc2)
                    collections_to_draw_now = collections_to_draw.copy()
                    lc1_now = plot.create_line_collection(net, lines=affected_only_array, color='orange')
                    lc2_new = plot.create_line_collection(net, lines=total_failed, color='red')
                    collections_to_draw_now.append(lc1_now)
                    collections_to_draw_now.append(lc2_now)

            ax.set_aspect('equal')
            plot.draw_collections(collections_to_draw_now, ax=ax)
            ax.autoscale_view()
            
            plt.title(f"Hour {i}")
            plt.savefig(f"hour_{i}.png")
            plt.close(fig)

        # Create a list of filenames in the order they were saved
        filenames = [f"hour_{i}.png" for i in range(Num_hrs)]
        # Compile the images into a GIF
        with imageio.get_writer('Impact of Windstorm.gif', mode='I', fps=2, loop = 0) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
            for filename in filenames:
                os.remove(filename)

        from IPython.display import Image, display
        display(Image(filename='Impact of Windstorm.gif'))

    def _set_cp_lat_n(self, val):
        '''Set cp_lat_n'''
        self.data.W.contour.dlat = val

    def _set_cp_lon_n(self, val):
        '''Set cp_lon_n'''
        self.data.W.contour.dlon = val

    def _set_lim_lng_ws(self, val):
        '''Set lim_lng_ws'''
        self.data.W.event.lng = val

    def _set_lim_max_v_ws(self, val):
        '''Set lim_max_v_ws'''
        self.data.W.event.max_v = val

    def _set_lim_ttr(self, val):
        '''Set lim_ttr'''
        self.data.W.event.ttr = val

    def _set_lng_ws(self, val):
        '''Set lng_ws'''
        self.MC.W.lng = val

    def _set_cp_dis_aggregated(self, val):
        '''Set cp_dis_aggregated'''
        self.data.W.contour.dis = val

    def _set_cp_from_to(self, val):
        '''Set cp_from_to'''
        self.data.W.contour.from_to = val

    def _set_cp_lat(self, val):
        '''Set cp_lat'''
        self.data.W.contour.lat = val

    def _set_cp_lon(self, val):
        '''Set cp_lon'''
        self.data.W.contour.lon = val

    def _set_cp_num(self, val):
        '''Set cp_num'''
        self.data.W.contour.num = val

    def _set_lim_v_ws(self, val):
        '''Set lim_v_ws'''
        self.MC.W.v = val

    def _set_max_ws_yr(self, val):
        '''Set max_ws_yr'''
        self.data.W.event.max_yr = val

    def _set_num_hrs_yr(self, val):
        '''Set num_hrs_yr'''
        self.data.num_hrs_yr = val

    def _set_num_ws_total(self, val):
        '''Set num_ws_total'''
        self.MC.W.total = val

    def _set_num_ws_yr(self, val):
        '''Set num_ws_yr'''
        self.MC.W.num_yr = val

    def _set_mcs_yr(self, val):
        '''Set num_mcs_yr'''
        self.data.MC.trials = val

    def _set_frg_mu(self, val):
        '''Set frg_mu'''
        self.data.frg.mu = val
    
    def _set_frg_sigma(self, val):
        '''Set frg_sigma'''
        self.data.frg.sigma = val
    
    def _set_frg_thrd_1(self, val):
        '''Set frg_thrd_1'''
        self.data.frg.thrd_1 = val
    
    def _set_frg_thrd_2(self, val):
        '''Set frg_thrd_2'''
        self.data.frg.thrd_2 = val
    
    def _set_frg_shift_f(self, val):
        '''Set frg_shift_f'''
        self.data.frg.shift_f = val