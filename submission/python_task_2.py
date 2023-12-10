import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    G = nx.from_pandas_edgelist(df, 'id_start', 'id_end', ['distance'])

    # Initialize an empty DataFrame for the distance matrix
    distance_matrix = pd.DataFrame(index=df['id_start'].unique(), columns=df['id_end'].unique())
    # Iterate through all pairs of toll locations and calculate cumulative distances
    for start_id in distance_matrix.index:
        for end_id in distance_matrix.columns:
            #print(end_id)
            if start_id == end_id:
                # Diagonal values should be 0
                distance_matrix.loc[start_id, end_id] = 0
                #print(distance_matrix)
            else:
                try:
                    # Calculate the shortest path distance between start and end IDs
                    distance = nx.shortest_path_length(G, source=start_id, target=end_id, weight='distance')
                    distance_matrix.loc[start_id, end_id] = distance
                except nx.NetworkXNoPath:
                    # If no path exists, set the distance to NaN
                    distance_matrix.loc[start_id, end_id] = 0
                    

                    
    # Make the matrix symmetric
    distance_matrix = distance_matrix.combine_first(distance_matrix.T)

    return distance_matrix

df = pd.read_csv('dataset-3.csv')
result_matrix = calculate_distance_matrix(df)
print(result_matrix)

def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    unrolled_df = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])

    # Iterate through each row of the distance matrix
    for start_id in df.index:
        for end_id in df.columns:
            # Exclude same id_start to id_end combinations and add the combination and distance to the unrolled DataFrame
            if start_id != end_id:
                unrolled_df = unrolled_df.append({'id_start': start_id, 'id_end': end_id, 'distance': df.loc[start_id, end_id]}, ignore_index=True)

    return unrolled_df

result_unrolled = unroll_distance_matrix(result_matrix)
print(result_unrolled)


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    # Filter rows based on the reference value and Calculate the average distance for the reference value
    reference_rows = df[df['id_start'] == reference_value]
    average_distance = reference_rows['distance'].mean()

    # Calculate the threshold values
    lower_threshold = average_distance - 0.1 * average_distance
    upper_threshold = average_distance + 0.1 * average_distance

    filtered_rows = df[(df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold)]

    # Get unique values from the id_start column and sort them
    result_ids = sorted(filtered_rows['id_start'].unique())
    columns = ['id_start']
    df = pd.DataFrame(result_within_threshold, columns=columns)
    # int_list = [int(x) for x in result_ids]
    return df

reference_value = 1001424
result_within_threshold = find_ids_within_ten_percentage_threshold(result_unrolled, reference_value)
print(result_within_threshold)


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Create new columns for each vehicle type with default value 0
    for vehicle_type in rate_coefficients.keys():
        df[vehicle_type] = 0

    # Calculate toll rates based on distance and rate coefficients
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    df['id_start'] = df['id_start'].astype(int)
    df['id_end'] = df['id_end'].astype(int)
        
    return df

result_with_toll_rate = calculate_toll_rate(result_unrolled)
print(result_with_toll_rate)

def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    # Define time ranges for weekdays and weekends
    weekday_time_ranges = [(time(0, 0), time(10, 0)), (time(10, 0), time(18, 0)), (time(18, 0), time(23, 59, 59))]
    weekend_time_ranges = [(time(0, 0), time(23, 59, 59))]

    # Initialize empty lists to store start_day, start_time, end_day, end_time values
    start_day_list, start_time_list, end_day_list, end_time_list = [], [], [], []

    # Iterate through each unique (id_start, id_end) pair
    for index, row in df.iterrows():
        # Iterate through days of the week
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            # Iterate through time ranges
            for start_time, end_time in weekday_time_ranges:
                # Append values to the lists
                start_day_list.append(day)
                start_time_list.append(start_time)
                end_day_list.append(day)
                end_time_list.append(end_time)

        # For weekends
        for day in ['Saturday', 'Sunday']:
            for start_time, end_time in weekend_time_ranges:
                start_day_list.append(day)
                start_time_list.append(start_time)
                end_day_list.append(day)
                end_time_list.append(end_time)

    # Create separate DataFrames for weekdays and weekends
    weekday_df = pd.DataFrame({
        'start_day': start_day_list,
        'start_time': start_time_list,
        'end_day': end_day_list,
        'end_time': end_time_list
    })

    # Convert 'id_start' and 'id_end' columns to 'object' type
    
    df['row'] = df.index + 1
    #print(df)
    weekday_df['row'] = weekday_df.index + 1
    
    # Merge the time intervals DataFrame with the input DataFrame based on 'id_start'
    result_df = pd.merge(df, weekday_df, on=['row'], how='inner')

    # print(result_df)
    # Fill NaN values in 'start_day', 'start_time', 'end_day', 'end_time' with appropriate values
    result_df['start_day'] = result_df['start_day_y'].fillna('Unknown')
    result_df['start_time'] = result_df['start_time_y'].fillna(time(0, 0))
    result_df['end_day'] = result_df['end_day_y'].fillna('Unknown')
    result_df['end_time'] = result_df['end_time_y'].fillna(time(0, 0))

    #print(result_df['start_day'])
    
    # Define discount factors for each time range
    discount_factors = {
        (time(0, 0), time(10, 0)): 0.8,
        (time(10, 0), time(18, 0)): 1.2,
        (time(18, 0), time(23, 59, 59)): 0.8,
        (time(0, 0), time(23, 59, 59)): 0.7  # Constant discount factor for weekends
    }

    # Apply discount factors based on time ranges
    for time_range, discount_factor in discount_factors.items():
        rows_to_update = (
            (result_df['start_time'] >= time_range[0]) & (result_df['end_time'] <= time_range[1])
        )

        for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
            # Adjust column names to include suffixes if present
            moto_col = f'{vehicle_type}_x' if f'{vehicle_type}_x' in result_df.columns else f'{vehicle_type}'
            result_df.loc[rows_to_update, moto_col] *= discount_factor

    # Drop unnecessary columns
    result_df = result_df.drop(columns=['row','start_day_x', 'start_time_x', 'end_day_x', 'end_time_x','start_day_y', 'start_time_y', 'end_day_y', 'end_time_y'])

    return result_df

    
result_with_time_based_toll_rates = calculate_time_based_toll_rates(result_with_toll_rate)
print(result_with_time_based_toll_rates)
