import pandas as pd

# Load the uploaded Excel file
file_path = '2023-2024 Plan for Dept Heads.xlsx'
xls = pd.ExcelFile(file_path)

# Load the Fall 2023 and Spring 2024 tabs with header at the correct row
df_fall_2023 = pd.read_excel(xls, sheet_name='Fall 2023 (2231)', header=2)
df_spring_2024 = pd.read_excel(xls, sheet_name='Spring 2024', header=2)

# Drop columns starting with 'Unnamed:'
df_fall_2023 = df_fall_2023.loc[:, ~df_fall_2023.columns.str.startswith('Unnamed:')]
df_spring_2024 = df_spring_2024.loc[:, ~df_spring_2024.columns.str.startswith('Unnamed:')]

# Identify the columns to keep (second set of Days, Time Start, Time End)
columns_to_keep = ['Days', 'Time Start', 'Time End']

# Remove the first set of Days, Time Start, Time End
df_fall_2023 = df_fall_2023.drop(columns=columns_to_keep)
df_spring_2024 = df_spring_2024.drop(columns=columns_to_keep)

# Rename the columns to the proper names
df_fall_2023 = df_fall_2023.rename(columns={
    'Days.1': 'Days',
    'Time Start.1': 'Time Start',
    'Time End.1': 'Time End'
})
df_spring_2024 = df_spring_2024.rename(columns={
    'Days.1': 'Days',
    'Time Start.1': 'Time Start',
    'Time End.1': 'Time End'
})

# Drop any remaining duplicate columns
df_fall_2023 = df_fall_2023.loc[:, ~df_fall_2023.columns.duplicated()]
df_spring_2024 = df_spring_2024.loc[:, ~df_spring_2024.columns.duplicated()]

# Drop the specified columns
columns_to_drop = ['Yr Level/ Reqrmt', 'Course Attribute', 'Crdt', 'Estimated NEED', 'Estimated ELIGIBLE', 'Zoom Links (for Hybrid courses and students granted exceptions to attend online ONLY)']
df_fall_2023 = df_fall_2023.drop(columns=columns_to_drop, errors='ignore')
df_spring_2024 = df_spring_2024.drop(columns=columns_to_drop, errors='ignore')

# Remove leading and trailing white spaces from all string columns
df_fall_2023 = df_fall_2023.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df_spring_2024 = df_spring_2024.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# List of wanted departments
wanted_departments = [
    'COMPUTING', 'BS CIT', 'BS COMPUTING SECURITY', 'MATH/SCIENCE',
]

# Adjusted function to filter the rows based on department names and their courses
def filter_departments_corrected(df, departments, class_col, subject_col, cat_col, sect_col, course_name_col):
    # Initialize an empty DataFrame to store the filtered rows
    filtered_df = pd.DataFrame(columns=df.columns)
    add_row = False

    for _, row in df.iterrows():
        # Check if the row is a department header
        if pd.isna(row[class_col]) and pd.isna(row[subject_col]) and pd.isna(row[cat_col]) and pd.isna(row[sect_col]):
            if row[course_name_col] in departments:
                add_row = True
            else:
                add_row = False
        # Add rows that belong to the wanted departments
        if add_row:
            filtered_df = pd.concat([filtered_df, pd.DataFrame([row], columns=df.columns)], ignore_index=True)

    return filtered_df

# Apply the function to both dataframes with the corrected column names
df_fall_2023_filtered_corrected = filter_departments_corrected(df_fall_2023, wanted_departments, 'Class #', 'Subject', 'Cat#', 'Sect#', 'Course Name')
df_spring_2024_filtered_corrected = filter_departments_corrected(df_spring_2024, wanted_departments, 'Class #', 'Subject', 'Cat#', 'Sect#', 'Course Name')

# Reset index of the filtered dataframes to ensure proper indexing
df_fall_2023_filtered_corrected.reset_index(drop=True, inplace=True)
df_spring_2024_filtered_corrected.reset_index(drop=True, inplace=True)

# Remove rows with 'OL' in 'Time Start' or 'Time End' under MATH/SCIENCE
def remove_ol_rows(df):
    math_science_idx = df[df['Course Name'] == 'MATH/SCIENCE'].index[0]
    df_math_science = df.loc[math_science_idx:]

    ol_rows = df_math_science[(df_math_science['Time Start'] == 'OL') & (df_math_science['Time End'] == 'OL')].index
    df = df.drop(ol_rows)

    return df

df_fall_2023_filtered_corrected = remove_ol_rows(df_fall_2023_filtered_corrected)
df_spring_2024_filtered_corrected = remove_ol_rows(df_spring_2024_filtered_corrected)

# Fill missing Days, Time Start, or Time End values
def fill_missing_values(df):
    for idx, row in df.iterrows():
        if pd.isna(row['Days']) or pd.isna(row['Time Start']) or pd.isna(row['Time End']):
            class_number = row['Class #']
            matching_rows = df[df['Class #'] == class_number]
            if not matching_rows.empty:
                if pd.isna(row['Days']) and not matching_rows['Days'].dropna().empty:
                    df.at[idx, 'Days'] = matching_rows['Days'].dropna().values[0]
                if pd.isna(row['Time Start']) and not matching_rows['Time Start'].dropna().empty:
                    df.at[idx, 'Time Start'] = matching_rows['Time Start'].dropna().values[0]
                if pd.isna(row['Time End']) and not matching_rows['Time End'].dropna().empty:
                    df.at[idx, 'Time End'] = matching_rows['Time End'].dropna().values[0]
    return df

df_fall_2023_filtered_corrected = fill_missing_values(df_fall_2023_filtered_corrected)
df_spring_2024_filtered_corrected = fill_missing_values(df_spring_2024_filtered_corrected)

# Fill '""""' values with the value from the row above
def fill_empty_values_with_above(df):
    for column in df.columns:
        df[column] = df[column].replace('"', pd.NA)
    df = df.fillna(method='ffill')
    return df

# Apply forward fill to the entire DataFrame
def forward_fill(df):
    return df.fillna(method='ffill')

# Apply fill_empty_values_with_above first, then forward fill
df_fall_2023_filtered_corrected = fill_empty_values_with_above(df_fall_2023_filtered_corrected)
df_spring_2024_filtered_corrected = fill_empty_values_with_above(df_spring_2024_filtered_corrected)

df_fall_2023_filtered_corrected = forward_fill(df_fall_2023_filtered_corrected)
df_spring_2024_filtered_corrected = forward_fill(df_spring_2024_filtered_corrected)

# Remove 'hidden section' values
def remove_hidden_section(df):
    df = df.applymap(lambda x: "" if x == "hidden section" else x)
    return df

df_fall_2023_filtered_corrected = remove_hidden_section(df_fall_2023_filtered_corrected)
df_spring_2024_filtered_corrected = remove_hidden_section(df_spring_2024_filtered_corrected)

# Save the updated filtered data to new CSV files
csv_fall_2023_filtered_corrected_updated_v2 = 'Fall_2023_Filtered_Corrected_Updated_v2.csv'
csv_spring_2024_filtered_corrected_updated_v2 = 'Spring_2024_Filtered_Corrected_Updated_v2.csv'

df_fall_2023_filtered_corrected.to_csv(csv_fall_2023_filtered_corrected_updated_v2, index=False)
df_spring_2024_filtered_corrected.to_csv(csv_spring_2024_filtered_corrected_updated_v2, index=False)
