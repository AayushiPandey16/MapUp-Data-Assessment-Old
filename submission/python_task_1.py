import pandas as pd


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here
    df = pd.pivot_table(df, values='car', index='id_1', columns='id_2')
    # Set diagonal values to 0
    df.values[[range(len(df))]*2] = 0
    return df


df = pd.read_csv('dataset-1.csv')
#print(type(df_dataset_1))
result_car_matrix = generate_car_matrix(df)
print(result_car_matrix)



def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here
    conditions = [
        (df['car'] <= 15),
        (df['car'] > 15) & (df['car'] <= 25),
        (df['car'] > 25)
    ]
    choices = ['low', 'medium', 'high']
    df['car_type'] = pd.Series(np.select(conditions, choices, default='unknown'), dtype='category')

    # Calculate the count of occurrences for each car_type category
    type_count = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    type_count_sorted = {key: type_count[key] for key in sorted(type_count.keys())}

    return type_count_sorted

df = pd.read_csv('dataset-1.csv')
result_type_count = get_type_count(df)
print(result_type_count)


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
    bus_indexes = df[df['bus'] > 2 * (df['bus'].mean())].index.tolist()
    bus_indexes.sort()
    return bus_indexes

df = pd.read_csv('dataset-1.csv')
result_bus_indexes = get_bus_indexes(df)
print(result_bus_indexes)


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here
    selected_routes = df.groupby('route')['truck'].mean() > 7
    sorted_routes = selected_routes[selected_routes].index.tolist()
    return sorted_routes

df = pd.read_csv('dataset-1.csv')
result_filtered_routes = filter_routes(df)
print(result_filtered_routes)


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here
    # Create a copy of the input DataFrame to avoid modifying the original
    modified_df = matrix.copy()

    # Apply the specified logic to modify the values
    modified_df[modified_df > 20] *= 0.75
    modified_df[modified_df <= 20] *= 1.25

    # Round the values to 1 decimal place
    modified_df = modified_df.round(1)

    return modified_df


result_modified = multiply_matrix(result_car_matrix)
print(result_modified)

def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    # Convert timestamp columns to datetime format
    df['start_time'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_time'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Calculate the time difference for each row
    df['time_diff'] = df['end_time'] - df['start_time']

    # Define the expected time range for a complete 24-hour period
    expected_time_range = pd.to_timedelta('1 day')

    # Define the expected days of the week (Monday to Sunday)
    expected_days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Check if each (id, id_2) pair has incorrect timestamps
    completeness_series = (
        (df['time_diff'] != expected_time_range) | 
        (~df['start_time'].dt.day_name().isin(expected_days_of_week)) |
        (~df['end_time'].dt.day_name().isin(expected_days_of_week))
    )

    # Create a multi-index boolean series (id, id_2)
    completeness_series = completeness_series.groupby(['id', 'id_2']).any()

    return completeness_series

df_dataset_2 = pd.read_csv('dataset-2.csv')
result_completeness = check_time_completeness(df_dataset_2)
print(result_completeness)