import pandas as pd
import sqlite3
import os

def csv_to_sqlite(csv_path, db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

if __name__ == "__main__":
    cities = ['nyc', 'berlin', 'amsterdam', 'sydney', 'rome', 'tokyo', 'barcelona', 'brussels']
    for city in cities:
        csv_path = f'./datasets/{city}/{city}_airbnb_listings.csv'
        db_path = 'airbnb_listings.db'
        table_name = f'{city}_listings'
        csv_to_sqlite(csv_path, db_path, table_name)
        print(f'{city} data has been loaded into {table_name} table in {db_path}')
