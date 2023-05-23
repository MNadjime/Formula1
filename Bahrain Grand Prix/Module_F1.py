#MyModuleF1

import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fastf1.plotting 
from timple.timedelta import strftimedelta
fastf1.Cache.enable_cache('ff1Cache')
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import operator

"Analyse - Course"





def PositionByLap(List_df,session):
    
    """Retourne un Dataframe 
    
    Args:
        Session (DataFrame): Dataframe de la session étudiée
        DataFrame (list): Liste de Dataframe

            
    Vars:
        Drivers_info (list) : Liste contenant les informations des pilotes
        Pos_Dataframe (DataFrame) : Dataframe contenant les positions des pilotes pour chaque tour
        Lap_Dataframe (DataFrame) : Dataframe contenant les informations des pilotes pour chaque tour de la course
        
    Returns:
        Position_by_Lap (DataFrame): Position des pilotes pour chaque tour"""
    
    List_df[1].Time = List_df[1].Time.round('1min')
    List_df[0].Time = List_df[0].Time.round('1min')
    
    Drivers_info = [(i,j) for i,j in zip(session.drivers,session.laps.Driver.unique())]
    
    Pos_Dataframe = [List_df[1][List_df[1].Driver == a] for a,b in Drivers_info]
    Lap_Dataframe = [List_df[0][List_df[0].Driver == a] for a,b in Drivers_info]
    
    Position_by_Lap = pd.DataFrame()
    
    for i,j,k in zip(Pos_Dataframe,Lap_Dataframe,Drivers_info):
        
        i = i.loc[i.Time.isin(j.Time)].drop_duplicates(subset=['Time'],keep='first').reset_index(drop=True)
        i['Driver'] = k[1]
        i['LapNumber'] = np.arange(1,len(i)+1)
        Position_by_Lap = Position_by_Lap.append(i[['Position','Driver','LapNumber']])

        
    return Position_by_Lap






def GapToLeaderByLap(List_df,session):
    
    """Retourne un Dataframe 
    
    Args:
        Session (DataFrame): Dataframe de la session étudiée
        DataFrame (list): Liste de Dataframe
            
    Vars:
        Drivers_info (list) : Liste contenant les informations des pilotes
        Gap_Dataframe (DataFrame) : Dataframe contenant les écarts avec le leader en course
        Lap_Dataframe (DataFrame) : Dataframe contenant les informations des pilotes pour chaque tour de la course
        
    Returns:
        Gap_By_Lap (DataFrame): Ecart avec le leader de la course pour chaque tour"""
    
    List_df[1].Time = List_df[1].Time.round('1min')
    List_df[0].Time = List_df[0].Time.round('1min')
    
    Drivers_info = [(i,j) for i,j in zip(session.drivers,session.laps.Driver.unique())]
    
    Gap_Dataframe = [List_df[1][List_df[1].Driver == a] for a,b in Drivers_info]
    Lap_Dataframe = [List_df[0][List_df[0].Driver == a] for a,b in Drivers_info]
    
    Gap_By_Lap = pd.DataFrame({'LapNumber' : []})
    
    for i,j,k in zip(Gap_Dataframe,Lap_Dataframe,Drivers_info):
        
        i = i.loc[i.Time.isin(j.Time)].drop_duplicates(subset=['Time'],keep='first').reset_index(drop=True)
        delta = len(j) - len(i)
        if ('LAP 1' not in i.GapToLeader and delta == 1):
            empty_lines = pd.concat([pd.DataFrame([i.GapToLeader[0]], columns=['GapToLeader'])],ignore_index=True)
            i = pd.concat([empty_lines,i]).reset_index(drop=True)

        i.GapToLeader.replace(regex={'LAP' : 0,r'^1.L$' : 110, r'^2.L$' : 210,'1L':110},inplace=True)
        i.GapToLeader = i.GapToLeader.apply(lambda x : float(x.replace('+','')) if '+' in str(x) else x)
        data = pd.DataFrame({'LapNumber' : np.arange(1,len(i)+1), k[1] : i.GapToLeader})

        Gap_By_Lap = pd.merge(Gap_By_Lap,data,on='LapNumber',how='outer')

        Gap_By_Lap.fillna(300,inplace=True)
        
        Gap_By_Lap.set_index('LapNumber',inplace=True)
        
    return Gap_By_Lap







def regression(LapsByDriver,Driver):
    
    x = np.array(LapsByDriver.loc[LapsByDriver['Driver']==Driver]['LapNumber'])
    y = np.array(LapsByDriver.loc[LapsByDriver['Driver']==Driver]['LapTime'])
    
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    
    polynomial_features= PolynomialFeatures(degree=3)
    x_poly = polynomial_features.fit_transform(x)
 
    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)
    
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
    x_p, y_p = zip(*sorted_zip)
    
    return x_p, y_p 







" Analyse des Qualifications "







def map_graph(session,num_sectors):
    
    Laps = session.laps
    
    team = list(Laps.Team.unique())
    drivers = list(Laps.Driver.unique())
    
    telemetry = pd.DataFrame()
    
    for drv in drivers:
        
        drv_lap = Laps.pick_driver(drv).pick_fastest()
        
        drv_tel = drv_lap.get_telemetry().add_distance()
        
        drv_tel['Driver'] = drv
        
        drv_tel['Team'] = drv_lap.Team
        
        telemetry = telemetry.append(drv_tel)
        
    telemetry = telemetry[['Driver','Distance','Team','Speed','X','Y']]
    
    total_distance = max(telemetry['Distance'])
    
    minisector_length = total_distance / num_sectors
    
    minisectors = [0]
    
    for i in range(0, (num_sectors - 1)):
        
        minisectors.append(minisector_length * (i + 1))
        
    telemetry['Minisector'] = telemetry['Distance'].apply(
        lambda z : (
            minisectors.index(
                min(minisectors,key=lambda x: abs(x-z)))+1
        )
    )
    
    avg_speed = telemetry.groupby(['Team', 'Minisector'])['Speed'].mean().reset_index()

    team_sectors = avg_speed.loc[avg_speed.groupby(['Minisector'])['Speed'].idxmax()]

    team_sectors = team_sectors[['Minisector','Team']].rename(columns = {'Team' : 'Team_Sect'})
    
    telemetry = telemetry.merge(team_sectors, on=['Minisector'])

    telemetry = telemetry.sort_values(by=['Distance'])

    telemetry_f = telemetry[['Distance','Speed','X','Y','Minisector','Team_Sect']]

    telemetry_f['Color'] = telemetry_f.loc[:,'Team_Sect'].apply(lambda x : fastf1.plotting.team_color(x))
    
    return telemetry_f







def plot_fastest_team(df,session):

    #PLOT QUALIFYING SESSION

    team_colors = list()
    for i in df.Team:
        color = fastf1.plotting.team_color(i)
        team_colors.append(color)

    fig, ax = plt.subplots(figsize=(15,12))
    bars = ax.barh(df.index, df['LapTimeDelta'],
            color=team_colors, edgecolor='grey')
    ax.set_yticks(df.index)
    ax.set_yticklabels(df['Team'])

    # show fastest at the top
    ax.invert_yaxis()
    # draw vertical lines behind the bars
    ax.set_axisbelow(True)
    #ax.xaxis.grid(False, which='major', linestyle='--', color='black', zorder=-1000)

    pole_lap_str = strftimedelta(df['LapTime'][0], '%m:%s.%ms')

    plt.suptitle(f"{session.event['EventName']} {session.event.year} Qualifying\n"
                 f"Fastest Lap: {pole_lap_str} ({df.Team[0]})",fontsize='x-large')


    ax.bar_label(bars, padding = 5)
    
    plt.show()

    
    
    
    
    
    
    
    
def plot_fastest_driver(df,session):
    
    ideal_lap = (df.Sector1Time.min()+df.Sector2Time.min()+df.Sector3Time.min())
    team_colors = [fastf1.plotting.team_color(team) for team in df.Team]
    
    fig,ax = plt.subplots(figsize=(15,10))
    bars = ax.barh(df.index, df['LapTimeDelta'], color = team_colors, edgecolor='grey')
    ax.set_yticks(df.index)
    ax.set_yticklabels(df.Driver)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    pole_lap_str = strftimedelta(df['LapTime'][0], '%m:%s.%ms')
    ideal_lap_str = strftimedelta(ideal_lap, '%m:%s.%ms')
    
    plt.suptitle(f"{session.event['EventName']} {session.event.year} Qualifying\n"
             f"Fastest Lap: {pole_lap_str} ({df['Driver'][0]})\n"
            f"Ideal Lap: {ideal_lap_str}",fontsize='x-large')
    ax.bar_label(bars, padding = 5)
    
    plt.show()


    
    
    
    
    
    
    
def Qualy_driver(session):
    
    
    """Retourne un dataframe contenant les détails des meilleurs tours pour chaque pilote et leur écart avec la pole position
    
    Args:
        session (DataFrame): DataFrame contenant les informations de la session étudiée
            
    Vars: 
        drivers (list): Liste des noms des pilotes
        pole_position (DataFrame): DataFrame du meilleur tour de la session étudiée
        
    Returns:
        Fastest_Lap (DataFrame): DataFrame contenant les détails des tours les plus rapides pour chaque pilote lors de la session 
        étudiée"""
    drivers = pd.unique(session.laps['Driver']) 
    
    list_fastest_lap = list()
    
    for drv in drivers : 
        
        drv_fastest_lap = session.laps.pick_driver(drv).pick_fastest()
        
        drv_fastest_lap['Driver'] = drv
        
        list_fastest_lap.append(drv_fastest_lap)
        
        
    Fastest_Lap = pd.DataFrame(list_fastest_lap).sort_values(by='LapTime').reset_index(drop=True)
    
    pole_position = session.laps.pick_fastest()
    
    Fastest_Lap['LapTimeDelta'] = Fastest_Lap['LapTime'] - pole_position['LapTime']
    
    Fastest_Lap['LapTimeDelta'] = Fastest_Lap.loc[:,'LapTimeDelta'].apply(lambda x : x.total_seconds())
    
    return Fastest_Lap










def Qualy_team(session):
    
    """Retourne un dataframe contenant les détails des meilleurs tours pour chaque écurie et leur écart avec l'écurie la plus 
    rapide
    
    Args:
        session (DataFrame): DataFrame contenant les informations de la session étudiée
            
    Vars: 
        fast_lap (DataFrame): DataFrame contenant les détails du tour le plus rapide de chaque pilote lors de la session étudiée
        
    Returns:
        avg_team (DataFrame): Temps au tour moyen de chaque écurie lors de la session étudiée"""
    
    fast_lap = Qualy_driver(session)
    
    avg_team = pd.DataFrame(fast_lap.groupby(['Team'])['LapTime'].mean()).sort_values(by='LapTime').reset_index()
    
    avg_team['LapTimeDelta'] = avg_team['LapTime'] - avg_team['LapTime'][0]
    
    avg_team['LapTimeDelta'] = avg_team.loc[:,'LapTimeDelta'].apply(lambda x:round(x.total_seconds(),3))
    
    return avg_team











def Gap(session,Driver):
    
    """Retourne un DataFrame 
    
    Args:
        session (DataFrame): DataFrame de la session étudiée
            
    Vars: 
        drivers (list): Liste des noms des pilotes
        pole_position (DataFrame): DataFrame du meilleur tour de la session
        
    Returns:
        Fastest_Lap (DataFrame): Meilleur tour de chaque pilote et son écart avec la pole position"""
    
    
    # Appel du tour référence et du meilleur tour du pilote sélectionné #
    
    lap_ref = session.laps.pick_fastest().get_car_data().add_relative_distance()
    
    best_lap_driver = session.laps.pick_driver(Driver).pick_fastest().get_car_data().add_relative_distance()
    
    # Transformation #
    
    lap_ref['Time'] = lap_ref.loc[:,'Time'].apply(lambda x : x.total_seconds())
    
    lap_ref.RelativeDistance = round(lap_ref.RelativeDistance,2)
    
    best_lap_driver['Time'] = best_lap_driver.loc[:,'Time'].apply(lambda x : x.total_seconds())
    
    best_lap_driver.RelativeDistance = round(best_lap_driver.RelativeDistance,2)
    
    
    best_lap_driver.drop_duplicates(subset=['RelativeDistance'], keep = 'first', inplace=True)
    
    lap_ref.drop_duplicates(subset=['RelativeDistance'], keep = 'first', inplace=True)
    
    # Alignement des deux Tours #
    
    if len(best_lap_driver)>=len(lap_ref):
        
        pts_idx = lap_ref.RelativeDistance
        
        best_lap_driver = best_lap_driver.loc[best_lap_driver['RelativeDistance'].isin(pts_idx)]
        
    else : 
        
        pts_idx = best_lap_driver.RelativeDistance
        
        lap_ref = lap_ref.loc[lap_ref['RelativeDistance'].isin(pts_idx)]
    
    return lap_ref.reset_index(drop=True), best_lap_driver.reset_index(drop=True)










def dict_colors(session):
    
    Colors = dict()
    for i in session.laps.Driver.unique():
        if i not in ['SAR','DEV']:
            Colors[i] = fastf1.plotting.driver_color(i)
        else:
            if i == 'SAR':
                Colors[i] = fastf1.plotting.driver_color('ALB')
            else:
                Colors[i] = fastf1.plotting.driver_color('TSU')
                
    return Colors








def main():

    print("niaw")


if __name__ == '__main__':

    main()