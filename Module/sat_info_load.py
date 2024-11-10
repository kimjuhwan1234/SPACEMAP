import re
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from sgp4.api import Satrec
from sgp4.ext import invjday
from astropy.time import Time
from astropy import units as u
from sgp4.conveniences import jday
from astropy.coordinates import ITRS
from datetime import datetime, timedelta
from astropy.coordinates import TEME, CartesianDifferential, CartesianRepresentation

np.set_printoptions(precision=2)


def get_tle(sat_num, username, password):
    '''
    :param sat_num: 위성번호
    :return: KVN과 TLE 데이터
    '''
    # Space-Track API의 URL
    url = "https://www.space-track.org/ajaxauth/login"

    # 요청에 필요한 인증 정보
    payload = {"identity": username, "password": password}

    # Session 객체 생성
    session = requests.Session()

    # 로그인 요청
    response = session.post(url, data=payload)
    if response.status_code != 200:
        print("로그인에 실패했습니다.")
        return None

    # KVN 요청
    kvn_response = session.get(
        f"https://www.space-track.org/basicspacedata/query/class/gp_history/NORAD_CAT_ID/{sat_num}/orderby/EPOCH%20ASC/EPOCH/2022-12-31--2024-08-01/format/kvn"
    )
    # TLE 요청
    tle_response = session.get(
        f"https://www.space-track.org/basicspacedata/query/class/gp_history/NORAD_CAT_ID/{sat_num}/orderby/EPOCH%20ASC/EPOCH/2022-12-31--2024-08-01/format/3le"
    )

    return kvn_response, tle_response


def compute_change_df(kvn_response, tle_response, real_time: bool):
    '''
    :param kvn_response: created time data
    :param tle_response: tle data
    :param real_time: TLE data sorting 기준 ex) created time or epoch time
    :return: TLE 정보가 담긴 DataFrame.
    '''

    # TLE DataFrame 생성
    tle_datas = tle_response.text.split('\r\n')
    if real_time:
        creation_dates = list(re.findall(r"CREATION_DATE\s*=\s*([\d\-T:]+)", kvn_response.text))
        info_df = pd.DataFrame([creation_dates, tle_datas[1:][::3], tle_datas[2:][::3]]).T
        info_df.columns = ['creation_date', 'first_line', 'second_line']
        info_df['created_year'] = info_df['creation_date'].map(lambda x: int(x.split('-')[0]))
        info_df['created_month'] = info_df['creation_date'].map(lambda x: int(x.split('-')[1]))
        info_df['created_day'] = info_df['creation_date'].map(lambda x: int(x.split('-')[2].split('T')[0]))
        info_df['created_hour'] = info_df['creation_date'].map(lambda x: int(x.split('T')[1].split(':')[0]))
        info_df['created_minute'] = info_df['creation_date'].map(lambda x: int(x.split(':')[1]))
        info_df['created_second'] = info_df['creation_date'].map(lambda x: int(x.split(':')[2]))
        info_df = info_df.reset_index(drop=True)

    if not real_time:
        info_df = pd.DataFrame([tle_datas[1:][::3], tle_datas[2:][::3]]).T
        info_df.columns = ['first_line', 'second_line']
        info_df = info_df.reset_index(drop=True)

    # epoch time 생성
    change_times = []
    for i in range(len(info_df)):
        tle_1, tle_2 = info_df['first_line'].iloc[i], info_df['first_line'].iloc[i]
        satellite = Satrec.twoline2rv(tle_1, tle_2)
        jdsatepoch = satellite.jdsatepoch
        jdsatepochfrac = satellite.jdsatepochF
        epochdatetime = invjday(jdsatepoch + jdsatepochfrac)
        change_times.append(epochdatetime)

    change_df = pd.DataFrame(change_times)
    change_df.columns = ['year', 'month', 'day', 'hour', 'minute', 'second']
    change_df.insert(0, 'first_line', info_df['first_line'])
    change_df.insert(1, 'second_line', info_df['second_line'])
    change_df['epoch_date'] = pd.to_datetime(change_df[['year', 'month', 'day', 'hour', 'minute', 'second']])

    # TLE 중복값 제거 및 sorting
    if real_time:
        change_df = pd.concat([change_df, info_df[
            ['created_year', 'created_month', 'created_day', 'created_hour', 'created_minute',
             'created_second', 'creation_date']]], axis=1)

        change_df = change_df.sort_values(by=['creation_date'])
        change_df = change_df.loc[
            change_df[['year', 'month', 'day', 'hour', 'minute']].drop_duplicates(keep='first').index].copy()
        change_df.reset_index(inplace=True, drop=True)
        change_df['real_time'] = 1

    if not real_time:
        change_df = change_df.sort_values(by=['epoch_date'])
        change_df = change_df.loc[
            change_df[['year', 'month', 'day', 'hour', 'minute']].drop_duplicates(keep='first').index].copy()
        change_df.reset_index(inplace=True, drop=True)

        end_point = change_df.iloc[-1:].copy()
        end_point['month'] = 8
        end_point['day'] = 1
        end_point['hour'] = 0
        end_point['minute'] = 0
        end_point['second'] = 0
        end_point['epoch_date'] = pd.to_datetime(end_point[['year', 'month', 'day', 'hour', 'minute', 'second']])
        change_df = pd.concat([change_df, end_point], axis=0).copy()
        change_df.reset_index(inplace=True, drop=True)
        change_df['real_time'] = 0

    return change_df


def compute_result_df(change_df: pd.DataFrame()):
    '''
    :param change_df: TLE 정보가 담긴 DataFrame.
    :return: 분당 통계량이 담긴 DataFrame.
    '''
    earth_radius = 6378.137
    rad2deg = 180.0 / np.pi

    # TLE 사이사이 마다 통계량 생성
    all_info = []
    for i in tqdm(range(len(change_df) - 1)):
        sat_info = change_df.iloc[i]
        next_sat_info = change_df.iloc[i + 1]

        tle_1, tle_2 = sat_info['first_line'], sat_info['second_line']
        satellite = Satrec.twoline2rv(tle_1, tle_2)
        jdsatepoch = satellite.jdsatepoch
        jdsatepochfrac = satellite.jdsatepochF
        epochdatetime = list(invjday(jdsatepoch + jdsatepochfrac))
        epochdatetime[-1] = int(epochdatetime[-1])

        next_tle_1, next_tle_2 = next_sat_info['first_line'], next_sat_info['second_line']
        next_satellite = Satrec.twoline2rv(next_tle_1, next_tle_2)
        next_jdsatepoch = next_satellite.jdsatepoch
        next_jdsatepochfrac = next_satellite.jdsatepochF
        next_epochdatetime = list(invjday(next_jdsatepoch + next_jdsatepochfrac))
        next_epochdatetime[-1] = int(next_epochdatetime[-1])

        if change_df['real_time'].all() == 1:
            start = (pd.to_datetime(sat_info['creation_date'])).to_pydatetime()
            end = (pd.to_datetime(next_sat_info['creation_date'])).to_pydatetime()
            end += timedelta(seconds=1)
            start += timedelta(minutes=1)
            start -= timedelta(seconds=start.second)

        if change_df['real_time'].all() == 0:
            start = datetime(*epochdatetime)
            if i == len(change_df) - 2:
                next_epochdatetime = [2024, 8, 1, 0, 0, 0]
                end = datetime(*next_epochdatetime)
            else:
                end = datetime(*next_epochdatetime)
                end += timedelta(seconds=1)
            start += timedelta(minutes=1)
            start -= timedelta(seconds=start.second)

        jd_lst = []
        fr_lst = []
        time_lst = []

        epoch = start
        while (epoch < end):
            year = epoch.year
            month = epoch.month
            date = epoch.day
            hour = epoch.hour
            minute = epoch.minute
            second = epoch.second
            jd, fr = jday(year, month, date, hour, minute, second)
            time_lst.append(epoch)
            jd_lst.append(jd)
            fr_lst.append(fr)
            epoch += timedelta(minutes=1)

        if len(jd_lst) == 0:
            continue

        e, r, v = satellite.sgp4_array(np.array(jd_lst), np.array(fr_lst))
        t_lst = Time(list(np.array(jd_lst) + np.array(fr_lst)), format='jd')
        teme_p = CartesianRepresentation(r[:, 0] * u.km, r[:, 1] * u.km, r[:, 2] * u.km)
        teme_v = CartesianDifferential(v[:, 0] * u.km / u.s, v[:, 1] * u.km / u.s, v[:, 2] * u.km / u.s)
        teme = TEME(teme_p.with_differentials(teme_v), obstime=t_lst)
        itrs_geo = teme.transform_to(ITRS(obstime=t_lst))
        locations = itrs_geo.earth_location
        geodetic_coords = locations.geodetic

        all_info.append(pd.DataFrame([time_lst,
                                      r[:, 0], r[:, 1], r[:, 2],
                                      v[:, 0], v[:, 1], v[:, 2],
                                      (r[:, 0] ** 2 + r[:, 1] ** 2 + r[:, 2] ** 2) ** 0.5,
                                      (v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2) ** 0.5,
                                      teme_p.x.value, teme_p.y.value, teme_p.z.value,
                                      teme_v.d_x.value, teme_v.d_y.value, teme_v.d_z.value,
                                      [satellite.alta * earth_radius] * len(time_lst),
                                      [satellite.altp * earth_radius] * len(time_lst),
                                      [satellite.inclo * rad2deg] * len(time_lst),
                                      [satellite.ecco] * len(time_lst),
                                      [satellite.nodeo * rad2deg] * len(time_lst),
                                      geodetic_coords.lon.value,
                                      geodetic_coords.lat.value,
                                      geodetic_coords.height.value,
                                      ]))

    # 최종 DataFrame 저장
    result = pd.concat(all_info, axis=1).T
    result.columns = ['Time', 'x', 'y', 'z', 'vx', 'vy', 'vz',
                      'Altitude', 'Velocity', 'x_earth', 'y_earth', 'z_earth',
                      'vx_earth', 'vy_earth', 'vz_earth', 'Apogee', 'Perigee', 'Inclination(deg)',
                      'Eccentricity', 'RAAN(deg)', 'Longitude', 'Latitude', 'Height']
    result['Time'] = pd.to_datetime(result['Time'])
    result = result.set_index(['Time'])
    return result


def load_data(file_path: str, sat_no: str, start_date: str, end_date: str, created_time: bool):
    '''
    :param file_path: 위성데이터가 있는 폴더경로
    :param sat_no: 위성번호
    :param start_date: 불러올 데이터의 시작날짜
    :param end_date: 불러올 데이터의 종료날짜
    :param created_time: created time인지 epoch time인지 여부
    :return: 시간 단위로 추출된 통계량 DataFrame
    '''
    if created_time:
        info_df = pd.read_csv(f'{file_path}/{sat_no}_created.csv')
    if not created_time:
        info_df = pd.read_csv(f'{file_path}/{sat_no}_epoch.csv')
    info_df = info_df.set_index('Time').sort_index()
    info_df.index = pd.to_datetime(info_df.index)
    info_df = info_df[info_df.index > start_date].copy()
    info_df = info_df[info_df.index <= end_date].copy()
    hour_df = info_df[::60].copy()
    hour_df['Semi-Major Axis(km)'] = (hour_df['Perigee'] + hour_df['Apogee']) / 2
    hour_df['Semi-Major Axis(km)'] = hour_df['Semi-Major Axis(km)'] / 100
    return hour_df
