# build_json.py
# A unified, parameter-driven script to build nested JSONL files from various relational datasets.
# Final version with all 7 builders fully implemented and support for custom I/O paths.

import pandas as pd
from pathlib import Path
import orjson  # Using orjson for better performance on large datasets
import argparse
from typing import Dict, Any, List, Iterator, Optional
from pandas import Timestamp
import numpy as np
from tqdm import tqdm
import warnings

# --- Global Settings & Warnings ---
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None  # default='warn'

# --- Utility Functions ---

def make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert non-JSON serializable types to compatible formats.
    Handles pandas/numpy data types and NaT/NaN values robustly.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(elem) for elem in obj]
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    if pd.isna(obj) or obj is pd.NaT:
        return None
    return obj

def save_to_jsonl(records: Iterator[Dict], output_path: Path):
    """
    Stream records to a JSONL file, showing progress.
    """
    print(f"\nStreaming records to JSONL file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    records_list = list(records)
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in tqdm(records_list, desc="Writing to JSONL"):
            cleaned_record = make_json_serializable(record)
            json_line = orjson.dumps(cleaned_record, option=orjson.OPT_NON_STR_KEYS).decode('utf-8')
            f.write(json_line + '\n')
            count += 1

    print(f"Successfully saved {count} records to {output_path}")

# --- Base Builder Class ---

class BaseBuilder:
    """
    Abstract base class for database-specific JSON builders.
    Now supports custom input and output paths.
    """
    def __init__(self, db_name: str, root_entity: str, db_path: Optional[str] = None, output_dir: Optional[str] = None):
        self.db_name = db_name
        self.root_entity = root_entity

        # --- PATH LOGIC: Use custom path if provided, otherwise use default ---
        if db_path:
            self.db_path = Path(db_path)
        else:
            self.db_path = Path.home() / f".cache/relbench/{self.db_name}/db"

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path.home() / f"relbench-data/{self.db_name}"
        # --- END OF PATH LOGIC ---

        self.prepared_data = None
        
        print(f"Initializing builder for database='{self.db_name}', root_entity='{self.root_entity}'")
        print(f"  - Input DB Path: {self.db_path}")
        print(f"  - Output Directory: {self.output_dir}")
        
        if not self.db_path.exists():
            print(f"WARNING: Database path does not exist: {self.db_path}")

    def load_and_prepare_data(self):
        """Load and preprocess data. Must be implemented by subclasses."""
        raise NotImplementedError

    def build_trees(self) -> Iterator[Dict]:
        """Build the nested dictionary for each root entity. Must be implemented by subclasses."""
        raise NotImplementedError

    def run(self):
        """Execute the full pipeline: prepare, build, and save."""
        print(f"\n--- Starting data preparation for {self.db_name} ---")
        self.load_and_prepare_data()
        if self.prepared_data is None:
            print("Data preparation failed. Exiting.")
            return

        print(f"\n--- Starting tree construction for root entity: {self.root_entity} ---")
        trees_iterator = self.build_trees()

        # The output filename is now constructed based on the final output_dir
        output_file = self.output_dir / f"{self.db_name}-{self.root_entity}.jsonl"
        save_to_jsonl(trees_iterator, output_file)

# --- Database-Specific Builders (Alphabetical Order & Fully Implemented) ---
# Note: The internal logic of these classes does not need to change, as they
#       rely on self.db_path which is set correctly in the BaseBuilder.

class AmazonBuilder(BaseBuilder):
    def load_and_prepare_data(self):
        try:
            print("Loading and merging parquet files for rel-amazon...")
            df_customer = pd.read_parquet(self.db_path / "customer.parquet")
            df_review = pd.read_parquet(self.db_path / "review.parquet")
            df_product = pd.read_parquet(self.db_path / "product.parquet")
            
            merged_df = pd.merge(df_review, df_product, on='product_id', how='left')
            full_data = pd.merge(merged_df, df_customer, on='customer_id', how='left')
            self.prepared_data = {'full_data': full_data}
            print("Data preparation complete.")
        except FileNotFoundError as e:
            print(f"Error: Data file not found - {e}")
            self.prepared_data = None

    def build_trees(self) -> Iterator[Dict]:
        if self.root_entity == 'customer':
            return self._build_customer_trees()
        elif self.root_entity == 'product':
            return self._build_product_trees()
        else:
            raise ValueError(f"Unsupported root entity for rel-amazon: '{self.root_entity}'. Supported: 'customer', 'product'.")

    def _build_customer_trees(self) -> Iterator[Dict]:
        for customer_id, group in self.prepared_data['full_data'].groupby('customer_id'):
            if group.empty: continue
            
            customer_info = group.iloc[0]
            customer_node = {
                'customer_id': int(customer_id),
                'customer_name': str(customer_info['customer_name']),
                'reviews': []
            }
            
            for _, review_data in group.iterrows():
                product_node = {
                    'product_id': int(review_data['product_id']),
                    'title': str(review_data['title']),
                    'brand': str(review_data['brand']),
                    'price': float(review_data['price']),
                    'category': str(review_data['category']),
                    'description': str(review_data['description'])
                }
                review_node = {
                    'review_text': str(review_data['review_text']),
                    'summary': str(review_data['summary']),
                    'review_time': str(review_data['review_time']),
                    'rating': float(review_data['rating']),
                    'verified': str(review_data['verified']),
                    'product': product_node
                }
                customer_node['reviews'].append(review_node)
            yield customer_node

    def _build_product_trees(self) -> Iterator[Dict]:
        for product_id, group in self.prepared_data['full_data'].groupby('product_id'):
            if group.empty: continue

            product_info = group.iloc[0]
            product_node = {
                'product_id': int(product_id),
                'title': str(product_info['title']),
                'brand': str(product_info['brand']),
                'price': float(product_info['price']),
                'category': str(product_info['category']),
                'description': str(product_info['description']),
                'reviews': []
            }

            for _, review_data in group.iterrows():
                customer_node = {
                    'customer_id': int(review_data['customer_id']),
                    'customer_name': str(review_data['customer_name'])
                }
                review_node = {
                    'review_text': str(review_data['review_text']),
                    'summary': str(review_data['summary']),
                    'review_time': str(review_data['review_time']),
                    'rating': float(review_data['rating']),
                    'verified': str(review_data['verified']),
                    'customer': customer_node
                }
                product_node['reviews'].append(review_node)
            yield product_node

class AvitoBuilder(BaseBuilder):
    def load_and_prepare_data(self):
        tables = ["UserInfo", "AdsInfo", "Category", "Location", "SearchInfo", "SearchStream", "VisitStream", "PhoneRequestsStream"]
        data = {}
        try:
            print("Loading and preparing data for rel-avito...")
            for name in tables:
                data[name] = pd.read_parquet(self.db_path / f"{name}.parquet")
            
            self.prepared_data = {
                'users_dict': data['UserInfo'].set_index('UserID').to_dict('index'),
                'ads_dict': data['AdsInfo'].set_index('AdID').to_dict('index'),
                'locations_dict': data['Location'].set_index('LocationID').to_dict('index'),
                'categories_dict': data['Category'].set_index('CategoryID').to_dict('index'),
                'searches_dict': data['SearchInfo'].set_index('SearchID').to_dict('index'),
                'appearances_by_ad': data['SearchStream'].groupby('AdID'),
                'visits_by_ad': data['VisitStream'].groupby('AdID'),
                'phone_requests_by_ad': data['PhoneRequestsStream'].groupby('AdID'),
                'searches_by_user': data['SearchInfo'].groupby('UserID'),
                'search_results_by_search': data['SearchStream'].groupby('SearchID'),
                'visits_by_user': data['VisitStream'].groupby('UserID'),
                'phone_requests_by_user': data['PhoneRequestsStream'].groupby('UserID'),
            }
            print("Data preparation complete.")
        except FileNotFoundError as e:
            print(f"Error: Data file not found - {e}")
            self.prepared_data = None

    def build_trees(self) -> Iterator[Dict]:
        if self.root_entity == 'ad':
            return (self._build_single_ad_tree(ad_id) for ad_id in self.prepared_data['ads_dict'])
        elif self.root_entity == 'user':
            return (self._build_single_user_tree(user_id) for user_id in self.prepared_data['users_dict'])
        else:
             raise ValueError(f"Unsupported root entity for rel-avito: '{self.root_entity}'. Supported: 'ad', 'user'.")

    def _build_ad_node(self, ad_id: int) -> Dict:
        ad_info = self.prepared_data['ads_dict'].get(ad_id, {})
        ad_node = {"AdID": ad_id, **ad_info}
        ad_node['ad_location'] = self.prepared_data['locations_dict'].get(ad_info.get('LocationID'), {})
        ad_node['ad_category'] = self.prepared_data['categories_dict'].get(ad_info.get('CategoryID'), {})
        return ad_node

    def _build_single_ad_tree(self, ad_id: int) -> Dict:
        ad_info = self.prepared_data['ads_dict'].get(ad_id, {})
        ad_node = {"AdID": ad_id, **ad_info}
        ad_node['ad_location'] = self.prepared_data['locations_dict'].get(ad_info.get('LocationID'), {})
        ad_node['ad_category'] = self.prepared_data['categories_dict'].get(ad_info.get('CategoryID'), {})

        search_appearances_list = []
        try:
            ad_appearances_df = self.prepared_data['appearances_by_ad'].get_group(ad_id)
            for _, appearance_row in ad_appearances_df.iterrows():
                appearance_node = appearance_row.to_dict()
                search_info = self.prepared_data['searches_dict'].get(appearance_row['SearchID'], {})
                appearance_node['search_details'] = search_info
                if search_info:
                    appearance_node['user_who_searched'] = self.prepared_data['users_dict'].get(search_info.get('UserID'), {})
                search_appearances_list.append(appearance_node)
        except KeyError:
            pass
        ad_node['search_appearances'] = search_appearances_list
        
        visits_list = []
        try:
            ad_visits_df = self.prepared_data['visits_by_ad'].get_group(ad_id)
            for _, visit_row in ad_visits_df.iterrows():
                visit_node = visit_row.to_dict()
                visit_node['user_who_visited'] = self.prepared_data['users_dict'].get(visit_row['UserID'], {})
                visits_list.append(visit_node)
        except KeyError:
            pass
        ad_node['user_visits'] = visits_list
        
        requests_list = []
        try:
            ad_requests_df = self.prepared_data['phone_requests_by_ad'].get_group(ad_id)
            for _, request_row in ad_requests_df.iterrows():
                request_node = request_row.to_dict()
                request_node['user_who_requested'] = self.prepared_data['users_dict'].get(request_row['UserID'], {})
                requests_list.append(request_node)
        except KeyError:
            pass
        ad_node['phone_requests'] = requests_list
            
        return ad_node

    def _build_single_user_tree(self, user_id: int) -> Dict:
        user_info = self.prepared_data['users_dict'].get(user_id, {})
        user_node = {"UserID": user_id, **user_info}

        search_history_list = []
        try:
            user_searches_df = self.prepared_data['searches_by_user'].get_group(user_id)
            for _, search_row in user_searches_df.iterrows():
                search_event_node = search_row.to_dict()
                search_event_node['search_location'] = self.prepared_data['locations_dict'].get(search_row['LocationID'], {})
                search_event_node['search_category'] = self.prepared_data['categories_dict'].get(search_row['CategoryID'], {})
                
                results_shown_list = []
                try:
                    search_results_df = self.prepared_data['search_results_by_search'].get_group(search_row['SearchID'])
                    for _, result_row in search_results_df.iterrows():
                        result_node = result_row.to_dict()
                        result_node['ad_shown'] = self._build_ad_node(result_row['AdID'])
                        results_shown_list.append(result_node)
                except KeyError:
                    pass
                search_event_node['search_results_shown'] = results_shown_list
                search_history_list.append(search_event_node)
        except KeyError:
            pass
        user_node['search_history'] = search_history_list

        visits_list = []
        try:
            user_visits_df = self.prepared_data['visits_by_user'].get_group(user_id)
            for _, visit_row in user_visits_df.iterrows():
                visit_node = visit_row.to_dict()
                visit_node['ad_viewed'] = self._build_ad_node(visit_row['AdID'])
                visits_list.append(visit_node)
        except KeyError:
            pass
        user_node['ad_visits'] = visits_list

        requests_list = []
        try:
            user_requests_df = self.prepared_data['phone_requests_by_user'].get_group(user_id)
            for _, request_row in user_requests_df.iterrows():
                request_node = request_row.to_dict()
                request_node['ad_requested'] = self._build_ad_node(request_row['AdID'])
                requests_list.append(request_node)
        except KeyError:
            pass
        user_node['phone_requests'] = requests_list
            
        return user_node

class EventBuilder(BaseBuilder):
    def load_and_prepare_data(self):
        tables = ["users", "user_friends", "events", "event_attendees", "event_interest"]
        data = {}
        try:
            print("Loading and preparing data for rel-event...")
            for name in tables:
                data[name] = pd.read_parquet(self.db_path / f"{name}.parquet")
            
            events_df = data['events'].copy()
            feature_cols = [f'c_{i}' for i in range(1, 101)]
            existing_feature_cols = [col for col in feature_cols if col in events_df.columns]
            if existing_feature_cols:
                events_df['c_vector'] = events_df[existing_feature_cols].values.tolist()
                events_df.drop(columns=existing_feature_cols, inplace=True)
                print("- Aggregated event features c_1 to c_100 into 'c_vector'.")

            self.prepared_data = {
                'users_dict': data['users'].set_index('user_id').to_dict('index'),
                'events_dict': events_df.set_index('event_id').to_dict('index'),
                'friends_by_user': data['user_friends'].groupby('user'),
                'events_by_creator': events_df.groupby('user_id'),
                'attendees_by_event': data['event_attendees'].groupby('event'),
                'interest_by_event': data['event_interest'].groupby('event'),
                'attendees_by_user': data['event_attendees'].groupby('user_id'),
                'interest_by_user': data['event_interest'].groupby('user'),
            }
            print("Data preparation complete.")
        except FileNotFoundError as e:
            print(f"Error: Data file not found - {e}")
            self.prepared_data = None

    def build_trees(self) -> Iterator[Dict]:
        if self.root_entity != 'user':
            raise ValueError(f"Unsupported root entity for rel-event: '{self.root_entity}'. Only 'user' is supported.")
        return (self._build_single_user_tree(uid) for uid in self.prepared_data['users_dict'])

    def _build_single_user_tree(self, user_id: int) -> Dict:
        user_node = {"user_id": user_id, **self.prepared_data['users_dict'].get(user_id, {})}
        
        try:
            friends_series = self.prepared_data['friends_by_user'].get_group(user_id)['friend']
            user_node['friends'] = friends_series.dropna().tolist()
        except KeyError:
            user_node['friends'] = []

        events_created_list = []
        created_event_ids = set()
        try:
            created_events_df = self.prepared_data['events_by_creator'].get_group(user_id)
            for _, event_row in created_events_df.iterrows():
                event_id = event_row['event_id']
                created_event_ids.add(event_id)
                event_node = event_row.to_dict()
                try:
                    event_node['attendees'] = self.prepared_data['attendees_by_event'].get_group(event_id).to_dict('records')
                except KeyError:
                    event_node['attendees'] = []
                try:
                    event_node['interest_shown'] = self.prepared_data['interest_by_event'].get_group(event_id).to_dict('records')
                except KeyError:
                    event_node['interest_shown'] = []
                events_created_list.append(event_node)
        except KeyError:
            pass
        user_node['events_created'] = events_created_list

        attended_ids = set(self.prepared_data['attendees_by_user'].get_group(user_id)['event'].dropna()) if user_id in self.prepared_data['attendees_by_user'].groups else set()
        interested_ids = set(self.prepared_data['interest_by_user'].get_group(user_id)['event'].dropna()) if user_id in self.prepared_data['interest_by_user'].groups else set()
        engaged_event_ids = (attended_ids | interested_ids) - created_event_ids
        
        engagements = []
        for event_id in engaged_event_ids:
            engagement_details = {}
            if event_id in attended_ids:
                try:
                    attendee_record = self.prepared_data['attendees_by_event'].get_group(event_id)
                    user_attendee_record = attendee_record[attendee_record['user_id'] == user_id]
                    if not user_attendee_record.empty:
                        engagement_details.update(user_attendee_record.iloc[0].to_dict())
                except KeyError: pass
            if event_id in interested_ids:
                try:
                    interest_record = self.prepared_data['interest_by_event'].get_group(event_id)
                    user_interest_record = interest_record[interest_record['user'] == user_id]
                    if not user_interest_record.empty:
                        engagement_details.update(user_interest_record.iloc[0].to_dict())
                except KeyError: pass

            engagements.append({
                "user_engagement_details": engagement_details,
                "event": self.prepared_data['events_dict'].get(event_id, {})
            })
        user_node['event_engagements'] = engagements
        return user_node

class F1Builder(BaseBuilder):
    def load_and_prepare_data(self):
        tables = ["drivers", "races", "circuits", "results", "qualifying", "constructors", "standings", "status"]
        data = {}
        try:
            print("Loading and preparing data for rel-f1...")
            for name in tables:
                data[name] = pd.read_parquet(self.db_path / f"{name}.parquet")
            
            races_with_circuits = pd.merge(data['races'], data['circuits'], on='circuitId', how='left')

            self.prepared_data = {
                'drivers_dict': data['drivers'].set_index('driverId').to_dict('index'),
                'constructors_dict': data['constructors'].set_index('constructorId').to_dict('index'),
                'races_with_circuits_dict': races_with_circuits.set_index('raceId').to_dict('index'),
                'status_dict': data['status'].set_index('statusId').to_dict('index'),
                'results_by_driver': data['results'].groupby('driverId'),
                'qualifying_by_race_driver': data['qualifying'].set_index(['raceId', 'driverId']),
                'standings_by_race_driver': data['standings'].set_index(['raceId', 'driverId'])
            }
            print("Data preparation complete.")
        except FileNotFoundError as e:
            print(f"Error: Data file not found - {e}")
            self.prepared_data = None
            
    def build_trees(self) -> Iterator[Dict]:
        if self.root_entity != 'driver':
            raise ValueError(f"Unsupported root entity for rel-f1: '{self.root_entity}'. Only 'driver' is supported.")
        return (self._build_single_driver_tree(did) for did in self.prepared_data['drivers_dict'])
        
    def _build_single_driver_tree(self, driver_id: int) -> Dict:
        driver_node = {"driverId": driver_id, **self.prepared_data['drivers_dict'].get(driver_id, {})}
        seasons = {}

        try:
            driver_results_df = self.prepared_data['results_by_driver'].get_group(driver_id)
        except KeyError:
            driver_node['seasons'] = []
            return driver_node
        
        for _, result_row in driver_results_df.iterrows():
            race_id = result_row['raceId']
            race_info = self.prepared_data['races_with_circuits_dict'].get(race_id, {})
            year = race_info.get('year')
            if not year: continue
            
            if year not in seasons:
                seasons[year] = {"year": year, "races": []}
            
            race_node = {**race_info}
            performance_node = {}
            performance_node['constructor'] = self.prepared_data['constructors_dict'].get(result_row['constructorId'], {})
            
            try:
                q_res = self.prepared_data['qualifying_by_race_driver'].loc[(race_id, driver_id)]
                performance_node['qualifying'] = q_res.to_dict()
            except KeyError:
                performance_node['qualifying'] = None
            
            result_node = result_row.to_dict()
            status_info = self.prepared_data['status_dict'].get(result_row['statusId'], {})
            result_node['status'] = {"statusId": result_row['statusId'], **status_info}
            performance_node['result'] = result_node
            
            try:
                s_res = self.prepared_data['standings_by_race_driver'].loc[(race_id, driver_id)]
                performance_node['championship_standing'] = s_res.to_dict()
            except KeyError:
                performance_node['championship_standing'] = None
            
            race_node['driver_performance'] = performance_node
            seasons[year]['races'].append(race_node)
            
        driver_node['seasons'] = sorted(seasons.values(), key=lambda x: x['year'])
        return driver_node

class HMBuilder(BaseBuilder):
    def load_and_prepare_data(self):
        tables = ["customer", "article", "transactions"]
        data = {}
        try:
            print("Loading and preparing data for rel-hm...")
            for name in tables:
                data[name] = pd.read_parquet(self.db_path / f"{name}.parquet")
            
            self.prepared_data = {
                'customer_dict': data['customer'].set_index('customer_id').to_dict('index'),
                'article_dict': data['article'].set_index('article_id').to_dict('index'),
                'transactions_by_customer': data['transactions'].groupby('customer_id'),
                'transactions_by_article': data['transactions'].groupby('article_id')
            }
            print("Data preparation complete.")
        except FileNotFoundError as e:
            print(f"Error: Data file not found - {e}")
            self.prepared_data = None

    def build_trees(self) -> Iterator[Dict]:
        if self.root_entity == 'customer':
            return (self._build_single_customer_tree(cid) for cid in self.prepared_data['customer_dict'])
        elif self.root_entity == 'article':
            return (self._build_single_article_tree(aid) for aid in self.prepared_data['article_dict'])
        else:
            raise ValueError(f"Unsupported root entity for rel-hm: '{self.root_entity}'. Supported: 'customer', 'article'.")

    def _build_single_customer_tree(self, customer_id: str) -> Dict:
        customer_node = {"customer_id": customer_id, **self.prepared_data['customer_dict'].get(customer_id, {})}
        purchase_history = []
        try:
            trans_df = self.prepared_data['transactions_by_customer'].get_group(customer_id)
            for _, row in trans_df.iterrows():
                article_id = row['article_id']
                transaction_node = {
                    "transaction_date": row['t_dat'],
                    "price": row['price'],
                    "sales_channel_id": row['sales_channel_id'],
                    "article_purchased": self.prepared_data['article_dict'].get(article_id, {})
                }
                purchase_history.append(transaction_node)
        except KeyError:
            pass
        customer_node['purchase_history'] = purchase_history
        return customer_node

    def _build_single_article_tree(self, article_id: int) -> Dict:
        article_node = {"article_id": article_id, **self.prepared_data['article_dict'].get(article_id, {})}
        purchase_history = []
        try:
            trans_df = self.prepared_data['transactions_by_article'].get_group(article_id)
            for _, row in trans_df.iterrows():
                customer_id = row['customer_id']
                transaction_node = {
                    "t_dat": row['t_dat'],
                    "price": row['price'],
                    "sales_channel_id": row['sales_channel_id'],
                    "customer_who_purchased": self.prepared_data['customer_dict'].get(customer_id, {})
                }
                purchase_history.append(transaction_node)
        except KeyError:
            pass
        article_node['purchase_history'] = purchase_history
        return article_node

class StackBuilder(BaseBuilder):
    def load_and_prepare_data(self):
        tables = ['users', 'posts', 'comments', 'votes', 'badges', 'postHistory', 'postLinks']
        data = {}
        try:
            print("Loading and preparing data for rel-stack...")
            for table in tables:
                path = self.db_path / f"{table}.parquet"
                data[table] = pd.read_parquet(path)

            data['users'].rename(columns={'Id': 'UserId'}, inplace=True)
            data['posts'].rename(columns={'Id': 'PostId', 'OwnerUserId': 'UserId'}, inplace=True)
            data['comments'].rename(columns={'Id': 'CommentId', 'UserId': 'CommenterUserId'}, inplace=True)
            data['votes'].rename(columns={'Id': 'VoteId', 'UserId': 'VoterUserId'}, inplace=True)
            data['badges'].rename(columns={'Id': 'BadgeId'}, inplace=True)
            data['postHistory'].rename(columns={'Id': 'HistoryId', 'UserId': 'EditorUserId'}, inplace=True)
            data['postLinks'].rename(columns={'Id': 'LinkId'}, inplace=True)

            self.prepared_data = data
            print("Data preparation complete.")
        except FileNotFoundError as e:
            print(f"Error: Data file not found - {e}")
            self.prepared_data = None

    def build_trees(self) -> Iterator[Dict]:
        if self.root_entity == 'post':
            return self._build_post_trees()
        elif self.root_entity == 'user':
            return self._build_user_trees()
        else:
            raise ValueError(f"Unsupported root entity for rel-stack: '{self.root_entity}'. Supported: 'post', 'user'.")
            
    def _aggregate_to_list(self, df):
        return df.to_dict('records')

    def _build_post_trees(self) -> Iterator[Dict]:
        print("Building post trees using vectorized aggregation...")
        posts_df = self._get_aggregated_posts_df()
        return (row for row in posts_df.to_dict('records'))

    def _build_user_trees(self) -> Iterator[Dict]:
        print("Building user trees using two-stage vectorized aggregation...")
        users_df = self.prepared_data['users'].copy()
        
        # Stage 1: Get aggregated posts dataframe
        posts_agg_df = self._get_aggregated_posts_df()

        # Stage 2: Aggregate posts and other children to users
        # Aggregate posts to users
        posts_by_user = posts_agg_df.groupby('UserId').apply(self._aggregate_to_list).rename('posts')
        users_df = users_df.merge(posts_by_user, on='UserId', how='left')
        
        # Aggregate other direct children to users
        user_children = {
            'badges': ('UserId', self.prepared_data['badges']),
            'user_comments': ('CommenterUserId', self.prepared_data['comments']),
            'user_votes': ('VoterUserId', self.prepared_data['votes']),
        }
        for name, (key, df) in user_children.items():
            if not df.empty and key in df.columns:
                aggregated = df.groupby(key).apply(self._aggregate_to_list).rename(name)
                users_df = users_df.merge(aggregated, left_on='UserId', right_on=key, how='left')

        return (row for row in users_df.to_dict('records'))

    def _get_aggregated_posts_df(self) -> pd.DataFrame:
        posts_df = self.prepared_data['posts'].copy()
        post_children = {
            'comments': ('PostId', self.prepared_data['comments']),
            'votes': ('PostId', self.prepared_data['votes']),
            'post_history': ('PostId', self.prepared_data['postHistory']),
            'post_links': ('PostId', self.prepared_data['postLinks'])
        }
        for name, (key, df) in post_children.items():
             if not df.empty and key in df.columns:
                aggregated = df.groupby(key).apply(self._aggregate_to_list).rename(name)
                posts_df = posts_df.merge(aggregated, on=key, how='left')
        
        # Enrich with owner info after all aggregations to avoid column conflicts
        posts_df = pd.merge(posts_df, self.prepared_data['users'].add_suffix('_owner'), left_on='UserId', right_on='UserId_owner', how='left')
        return posts_df


class TrialBuilder(BaseBuilder):
    def load_and_prepare_data(self):
        tables = [
            'studies', 'designs', 'eligibilities', 'outcomes', 'outcome_analyses', 
            'drop_withdrawals', 'reported_event_totals', 'sponsors', 'facilities', 
            'conditions', 'interventions', 'sponsors_studies', 'facilities_studies', 
            'conditions_studies', 'interventions_studies'
        ]
        data = {}
        try:
            print("Loading and preparing data for rel-trial...")
            for name in tables:
                data[name] = pd.read_parquet(self.db_path / f"{name}.parquet")

            grouped_data = {
                'outcomes_by_nct': data['outcomes'].groupby('nct_id'),
                'analyses_by_outcome': data['outcome_analyses'].groupby('outcome_id'),
                'drops_by_nct': data['drop_withdrawals'].groupby('nct_id'),
                'events_by_nct': data['reported_event_totals'].groupby('nct_id')
            }
            
            many_to_many = {'sponsors': 'sponsor_id', 'facilities': 'facility_id', 'conditions': 'condition_id', 'interventions': 'intervention_id'}
            for name, join_key in many_to_many.items():
                merged = pd.merge(data[f"{name}_studies"], data[name], on=join_key, how='left')
                grouped_data[f'{name}_by_nct'] = merged.groupby('nct_id')

            self.prepared_data = {
                'studies': data['studies'],
                'designs': data['designs'].set_index('nct_id'),
                'eligibilities': data['eligibilities'].set_index('nct_id'),
                **grouped_data
            }
            print("Data preparation complete.")
        except FileNotFoundError as e:
            print(f"Error: Data file not found - {e}")
            self.prepared_data = None

    def build_trees(self) -> Iterator[Dict]:
        if self.root_entity != 'study':
            raise ValueError(f"Unsupported root entity for rel-trial: '{self.root_entity}'. Only 'study' is supported.")
        return (self._build_single_study_tree(row) for _, row in self.prepared_data['studies'].iterrows())

    def _build_single_study_tree(self, study_row: pd.Series) -> Dict:
        nct_id = study_row['nct_id']
        study_node = study_row.to_dict()

        try: study_node['design'] = self.prepared_data['designs'].loc[nct_id].to_dict()
        except KeyError: study_node['design'] = None
        try: study_node['eligibility'] = self.prepared_data['eligibilities'].loc[nct_id].to_dict()
        except KeyError: study_node['eligibility'] = None

        for name in ['sponsors', 'facilities', 'conditions', 'interventions']:
            try: study_node[name] = self.prepared_data[f'{name}_by_nct'].get_group(nct_id).to_dict('records')
            except KeyError: study_node[name] = []

        try:
            outcomes_df = self.prepared_data['outcomes_by_nct'].get_group(nct_id)
            outcomes_list = []
            for _, outcome_row in outcomes_df.iterrows():
                outcome_node = outcome_row.to_dict()
                outcome_id = outcome_node.get('id')
                if outcome_id and pd.notna(outcome_id):
                    try: outcome_node['analyses'] = self.prepared_data['analyses_by_outcome'].get_group(outcome_id).to_dict('records')
                    except KeyError: outcome_node['analyses'] = []
                else:
                    outcome_node['analyses'] = []
                outcomes_list.append(outcome_node)
            study_node['outcomes'] = outcomes_list
        except KeyError:
            study_node['outcomes'] = []

        try: study_node['drop_withdrawals'] = self.prepared_data['drops_by_nct'].get_group(nct_id).to_dict('records')
        except KeyError: study_node['drop_withdrawals'] = []
        try: study_node['reported_event_totals'] = self.prepared_data['events_by_nct'].get_group(nct_id).to_dict('records')
        except KeyError: study_node['reported_event_totals'] = []

        return study_node


# --- Dispatcher ---

BUILDER_REGISTRY = {
    'rel-amazon': AmazonBuilder,
    'rel-avito': AvitoBuilder,
    'rel-event': EventBuilder,
    'rel-f1': F1Builder,
    'rel-hm': HMBuilder,
    'rel-stack': StackBuilder,
    'rel-trial': TrialBuilder,
}

# --- Main Execution ---

def main():
    """Main function to parse arguments and run the specified builder."""
    parser = argparse.ArgumentParser(
        description="Unified JSON builder for relational datasets.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("db_name", type=str, choices=BUILDER_REGISTRY.keys(),
                        help="The name of the database to process.")
    parser.add_argument("root_entity", type=str,
                        help="The root entity for building the JSON trees (e.g., 'user', 'post', 'study').")
    parser.add_argument("--db_path", type=str, default=Path.home() / f".cache/relbench/{parser.parse_args().db_name}/db",
                        help="Custom path to the database directory (e.g., './my_data/rel-amazon/db').\n"
                             "If not provided, defaults to '~/.cache/relbench/{db_name}/db'.")
    parser.add_argument("--output_dir", type=str, default=Path.home() / f"relbench-data-test/{parser.parse_args().db_name}",
                        help="Custom path to the output directory (e.g., './my_output').\n"
                             "If not provided, defaults to '~/relbench-data/{db_name}'.")
    
    args = parser.parse_args()
    builder_class = BUILDER_REGISTRY.get(args.db_name)
    
    if not builder_class:
        print(f"Error: No builder found for database '{args.db_name}'.")
        return

    try:
        # Pass all arguments to the builder's constructor
        builder = builder_class(
            db_name=args.db_name,
            root_entity=args.root_entity,
            db_path=args.db_path,
            output_dir=args.output_dir
        )
        builder.run()
    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # For debugging, you might want to re-raise the exception:
        # raise

if __name__ == "__main__":
    main()