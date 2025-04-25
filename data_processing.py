import os
import numpy as np
import pandas as pd
import psycopg2

def get_numeric_columns(conn, table_name, schema='public'):
    """Returns a list of numeric columns from a PostgreSQL table."""
    query = f"""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = %s AND table_name = %s
      AND data_type IN ('integer', 'smallint', 'bigint', 'decimal', 'numeric', 'real', 'double precision');
    """
    with conn.cursor() as cur:
        cur.execute(query, (schema, table_name))
        result = cur.fetchall()
        return [row[0] for row in result]

def connect_to_database(connection_string):
    """Establish connection to PostgreSQL database."""
    try:
        conn = psycopg2.connect(connection_string)
        print("Successfully connected to PostgreSQL.")
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def fetch_product_data(connection_string, table_name='mock_data'):
    """Fetch car data from database and prepare it for embedding."""
    conn = None
    cur = None
    data = []
    
    try:
        conn = connect_to_database(connection_string)
        if not conn:
            return []
            
        # Create a cursor
        cur = conn.cursor()
        
        # Execute query to fetch all data
        sql_query = f"SELECT * FROM {table_name};"
        print(f"Executing query: {sql_query}")
        cur.execute(sql_query)
        
        # Fetch all results
        rows = cur.fetchall()
        print(f"Fetched {len(rows)} rows.")
        
        # Get column names and numeric columns
        column_names = [desc[0] for desc in cur.description]
        numeric_columns = get_numeric_columns(conn, table_name)
        print(f"Numeric columns are: {numeric_columns}")
        
        # Calculate quantiles for categorization
        prices = []
        mileages = []
        engine_sizes = []
        for row in rows:
            row_data = dict(zip(column_names, row))
            prices.append(float(row_data['price']))
            mileages.append(float(row_data['mileage']))
            engine_sizes.append(float(row_data['engine_size']))
        
        # Compute quantiles
        price_q1, price_q2 = np.percentile(prices, [33.0, 66.0])
        mileage_q1, mileage_q2 = np.percentile(mileages, [33.0, 66.0])
        engine_q1, engine_q2 = np.percentile(engine_sizes, [33.0, 66.0])
        
        # Process data and create descriptive texts
        for row in rows:
            row_data = dict(zip(column_names, row))
            price_cat = categorize_price(row_data['price'], price_q1, price_q2)
            mileage_cat = categorize_mileage(row_data['mileage'], mileage_q1, mileage_q2)
            engine_cat = categorize_engine_size(row_data['engine_size'], engine_q1, engine_q2)

            text = f"""A {row_data['color']} {row_data['car_year']} {row_data['car_make']} {row_data['car_model']} with a {engine_cat} sized {row_data['fuel_type']} engine, {row_data['transmission']} transmission, and travelled {mileage_cat} distance. Price is ${row_data['price']} which is in {price_cat} segment."""
            
            data.append({
                "id": row_data["id"],
                "car_make": row_data["car_make"],
                "car_model": row_data["car_model"],
                "car_year": row_data["car_year"],
                "mileage": mileage_cat,
                "price": price_cat,
                "color": row_data["color"],
                "fuel_type": row_data["fuel_type"],
                "transmission": row_data["transmission"],
                "engine_size": engine_cat,
                "text": text
            })
        
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("PostgreSQL connection closed.")
    
    return data

def categorize_price(price, q1, q2):
    """Categorize price into budget, midrange, or premium."""
    if price <= q1:
        return 'budget'
    elif price <= q2:
        return 'midrange'
    else:
        return 'premium'

def categorize_mileage(mileage, q1, q2):
    """Categorize mileage into small, medium, or large."""
    if mileage <= q1:
        return 'small'
    elif mileage <= q2:
        return 'medium'
    else:
        return 'large'

def categorize_engine_size(engine_size, q1, q2):
    """Categorize engine size into small, medium, or large."""
    if engine_size <= q1:
        return 'small'
    elif engine_size <= q2:
        return 'medium'
    else:
        return 'large'