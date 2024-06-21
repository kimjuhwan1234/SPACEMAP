import json
import datetime
import requests
import statistics
import numpy as np
from sgp4.api import Satrec
from sgp4.api import days2mdhms, jday


def get_spacemap_tles(time_dict: dict, norad_ids: list):
    '''
    :param time_dict: 시간정보가 있는 딕셔너리
    :param norad_ids: 5자리 정수가 str형태로 들어간 리스트
    :return: TLE 딕셔너리 -> key: id, values: 딕셔너리 형태 TLE정보 name, firstLine, secondLine이 key임.
    '''
    server_url = "https://platformapi.spacemap42.com"
    # SEVERURL = "http://localhost:8082"
    tles = {}
    for norad_id in norad_ids:
        url = f"{server_url}/tles/{time_dict['year']:02d}/{time_dict['month']:02d}/{time_dict['day']:02d}/{0:02d}?id={norad_id}"
        response = requests.get(url)
        result = json.loads(response.text)
        tles[norad_id] = result["data"]["tles"][0]

    return tles


def get_spacetrack_tles(username: str, password: str, norad_cat_ids: list):
    '''
    :param username: space-track 아이디: 이메일형식
    :param password: space-track 비밀번호
    :param norad_cat_ids: 5자리 정수가 str형태로 들어간 리스트
    :return: TLE 딕셔너리 -> key: id, values: tuple 형태 TLE정보
    '''
    # Space-Track API의 로그인 URL
    login_url = "https://www.space-track.org/ajaxauth/login"
    # 요청에 필요한 인증 정보
    payload = {"identity": username, "password": password}
    # Session 객체 생성
    session = requests.Session()
    # 로그인 요청
    response = session.post(login_url, data=payload)

    if response.status_code != 200:
        print("로그인에 실패했습니다.")
        print(f"상태 코드: {response.status_code}")
        return None

    # TLE 데이터 저장용 딕셔너리
    tles = {}

    # 각 NORAD Catalog ID에 대해 TLE 데이터 요청
    for norad_cat_id in norad_cat_ids:
        tle_url = f"https://www.space-track.org/basicspacedata/query/class/gp/NORAD_CAT_ID/{norad_cat_id}/format/tle/emptyresult/show"
        response = session.get(tle_url)

        tle_lines = response.text.strip().split('\n')
        if len(tle_lines) >= 2:
            tles[norad_cat_id] = (tle_lines[0], tle_lines[1])
        else:
            print(f"NORAD Catalog ID {norad_cat_id}의 TLE 데이터 형식이 올바르지 않습니다.")
            tles[norad_cat_id] = None

    return tles


def cal_satellite_period(first_line, second_line):
    '''
    :param first_line: TLE 첫번째 줄
    :param second_line: TLE 두번째 줄
    :return satrec 객체와 초단위 위성주기
    '''
    satellite = Satrec.twoline2rv(first_line, second_line)
    mean_motion = satellite.no_kozai
    mean_motion_in_degrees = mean_motion * 180 / np.pi
    period_of_satellite = 360 / mean_motion_in_degrees
    period_of_satellite_in_seconds = period_of_satellite * 60

    print(f"Radian: {mean_motion:.4f}")
    print(f"Degree: {mean_motion_in_degrees:.4f}")
    print(f"Orbital period(min): {period_of_satellite:.4f}")
    print(f"Orbital period(sec): {period_of_satellite_in_seconds:.4f}")

    return satellite, period_of_satellite_in_seconds


def cal_kinematics_vector_periods(satellite, period_of_satellite_in_seconds: float, check: bool):
    '''
    :param satellite: satrec객체
    :param period_of_satellite_in_seconds: 초단위 위성주기
    :param check: 결과를 출력할지 여부
    :return: position vector와 velocity vector 리스트
    '''
    # Get epoch time
    month, day, hour, minute, second = days2mdhms(satellite.epochyr, satellite.epochdays)
    epoch_datetime = datetime.datetime(2024, month, day, hour, minute, int(second))
    positions = []
    velocities = []

    for second in range(0, int(period_of_satellite_in_seconds), 60):
        time = epoch_datetime + datetime.timedelta(seconds=second)
        # Julian Date 변환 (Date -> Julian Date)
        jd, fr = jday(time.year, time.month, time.day, time.hour, time.minute, time.second)
        # 위성의 위치 계산
        error_code, position, velocity = satellite.sgp4(jd, fr)

        positions.append(position)
        velocities.append(velocity)
        l2_norm_position = (position[0] ** 2 + position[1] ** 2 + position[2] ** 2) ** 0.5
        if check:
            print(time, "S:", position, "V:", velocity)
            print(f"Altitude: {l2_norm_position:.4f}")

    return error_code, positions, velocities


def cal_kinematics_vector_point(time_dict: dict, first_line, second_line):
    '''
    :param time_dict: 시간정보가 있는 딕셔너리
    :param first_line: TLE 첫쨰줄
    :param second_line: TLE 둘째줄
    :return: satrec개체, error, position vector, 그리고 velocity vector 튜플
    '''
    satellite = Satrec.twoline2rv(first_line, second_line)
    jd, fr = jday(time_dict['year'], time_dict['month'], time_dict['day'], 0, 0, 0)
    e, position, velocity = satellite.sgp4(jd, fr)
    return satellite, e, position, velocity


def print_statistics(data: dict):
    '''
    :param data: 딕셔너리
    :return: None
    '''
    values = list(data.values())

    min_value = min(values)
    max_value = max(values)
    average_value = statistics.mean(values)
    median_value = statistics.median(values)
    std_dev = statistics.stdev(values)
    variance = statistics.variance(values)
    data_range = max_value - min_value

    print(f"min: {min_value}")
    print(f"max: {max_value}")
    print(f"average: {average_value}")
    print(f"median: {median_value}")
    print(f"standard deviation: {std_dev}")
    print(f"variance: {variance}")
    print(f"range: {data_range}\n")


def find_max_key(data):
    max_key = None
    max_value = 0
    for key, value in data.items():
        if value > max_value:
            max_key = key
            max_value = value
    return max_key
