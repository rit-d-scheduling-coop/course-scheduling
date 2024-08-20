import pandas as pd

def clean_excel_file(file_path, sheet_name):
    # Load the Excel file
    xls = pd.ExcelFile(file_path)
    
    # Load the specified sheet with header at the correct row
    df = pd.read_excel(xls, sheet_name=sheet_name, header=2)
    
    # Drop columns starting with 'Unnamed:'
    df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]
    
    # Identify the columns to keep (second set of Days, Time Start, Time End)
    columns_to_keep = ['Days', 'Time Start', 'Time End']
    
    # Remove the first set of Days, Time Start, Time End
    df = df.drop(columns=columns_to_keep, errors='ignore')
    
    # Rename the columns to the proper names
    df = df.rename(columns={
        'Days.1': 'Days',
        'Time Start.1': 'Time Start',
        'Time End.1': 'Time End'
    })
    
    # Rename 'Course Attribute' to 'Yr Level/Reqrmt' in the Spring 2024 sheet
    if sheet_name == 'Spring 2024':
        df = df.rename(columns={'Course Attribute': 'Yr Level/ Reqrmt'})
    
    # Drop any remaining duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Drop the specified columns
    columns_to_drop = ['Crdt', 'Estimated NEED', 'Estimated ELIGIBLE', 'Zoom Links (for Hybrid courses and students granted exceptions to attend online ONLY)']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Remove leading and trailing white spaces from all string columns
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    return df

def filter_departments(df, departments):
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

    filtered_df = filter_departments_corrected(df, departments, 'Class #', 'Subject', 'Cat#', 'Sect#', 'Course Name')
    
    return filtered_df.reset_index(drop=True)

def remove_ol_rows(df):
    math_science_idx = df[df['Course Name'] == 'MATH/SCIENCE'].index[0]
    df_math_science = df.loc[math_science_idx:]

    ol_rows = df_math_science[(df_math_science['Time Start'] == 'OL') & (df_math_science['Time End'] == 'OL')].index
    df = df.drop(ol_rows)

    return df

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

def fill_empty_values_with_above(df):
    # Identify department headers
    is_header = df.apply(lambda row: pd.isna(row['Class #']) and pd.isna(row['Subject']) and pd.isna(row['Cat#']) and pd.isna(row['Sect#']) and not pd.isna(row['Course Name']), axis=1)
    
    # Separate headers and non-headers
    headers = df[is_header].copy()
    non_headers = df[~is_header].copy()
    
    # Fill non-header rows
    for column in non_headers.columns:
        non_headers[column] = non_headers[column].replace('"', pd.NA)
    non_headers = non_headers.fillna(method='ffill')
    
    # Combine headers and filled non-headers
    df_combined = pd.concat([headers, non_headers]).sort_index().reset_index(drop=True)
    
    return df_combined

def remove_hidden_section(df):
    return df.applymap(lambda x: "" if x == "hidden section" else x)

def process_excel_file(file_path, sheet_name, wanted_departments):
    # Clean the file
    df = clean_excel_file(file_path, sheet_name)
    
    # Filter departments
    df_filtered = filter_departments(df, wanted_departments)
    
    # Remove 'OL' rows for MATH/SCIENCE
    df_filtered = remove_ol_rows(df_filtered)
    
    # Fill missing values
    df_filtered = fill_missing_values(df_filtered)
    
    # Fill '""""' values with the value from the row above
    df_filtered = fill_empty_values_with_above(df_filtered)
    
    # Remove 'hidden section' values
    df_filtered = remove_hidden_section(df_filtered)
    
    return df_filtered

def process_and_save_excel_file(file_path, sheet_name, wanted_departments = ['COMPUTING', 'BS CIT', 'BS COMPUTING SECURITY', 'MATH/SCIENCE']):

    df_cleaned = process_excel_file(file_path, sheet_name, wanted_departments)

    # Generate output path based on sheet name
    output_path = f"excel/{sheet_name.replace(' ', '_')}_Filtered_Corrected.csv"
    # Save the cleaned data
    df_cleaned.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Usage example
    file_path = 'excel/2023-2024 Plan for Dept Heads.xlsx'
    wanted_departments = ['COMPUTING', 'BS CIT', 'BS COMPUTING SECURITY', 'MATH/SCIENCE']
    sheet_name = 'Fall 2023 (2231)'  # or 'Spring 2024'
    output_path = 'excel/Fall_2023_Filtered.csv'

    process_and_save_excel_file(file_path, sheet_name)

    sheet_name = 'Spring 2024'
    output_path = 'excel/Spring_2024_Filtered.csv'

    process_and_save_excel_file(file_path, sheet_name)