import numpy as np
import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Unified 15-feature common core
COMMON_FEATURES = [
    'flow_duration',        # Duration of the flow (seconds)
    'total_fwd_packets',    # Forward packet count
    'total_bwd_packets',    # Backward packet count
    'src_bytes',            # Source bytes
    'dst_bytes',            # Destination bytes
    'avg_pkt_size',         # Average packet size
    'min_pkt_size',         # Minimum packet size
    'max_pkt_size',         # Maximum packet size
    'std_pkt_size',         # Std dev of packet sizes
    'syn_flag_count',       # SYN flag count
    'ack_flag_count',       # ACK flag count
    'fin_flag_count',       # FIN flag count
    'rst_flag_count',       # RST flag count
    'protocol_tcp',         # TCP protocol (binary)
    'protocol_udp'          # UDP protocol (binary)
]

# Unified 7-class attack taxonomy
UNIFIED_CLASSES = {
    0: 'Benign',
    1: 'DoS_DDoS',
    2: 'Probe_Recon',
    3: 'Brute_Force',
    4: 'Web_Attack',
    5: 'Botnet_Malware',
    6: 'Spoofing_MITM'
}


# NSL-KDD Processing


NSLKDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# NSL-KDD attack mapping to unified classes
NSLKDD_ATTACK_MAP = {
    # Benign
    'normal': 0,
    # DoS/DDoS
    'back': 1, 'land': 1, 'neptune': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,
    'apache2': 1, 'mailbomb': 1, 'processtable': 1, 'udpstorm': 1,
    # Probe/Recon
    'ipsweep': 2, 'nmap': 2, 'portsweep': 2, 'satan': 2, 'mscan': 2, 'saint': 2,
    # Brute Force
    'guess_passwd': 3, 'ftp_write': 3, 'warezmaster': 3, 'warezclient': 3,
    # Web Attack
    'phf': 4, 'sqlattack': 4, 'xterm': 4,
    # Botnet/Malware
    'buffer_overflow': 5, 'rootkit': 5, 'loadmodule': 5, 'perl': 5,
    'worm': 5, 'httptunnel': 5, 'ps': 5,
    # Spoofing/MITM
    'imap': 6, 'multihop': 6, 'spy': 6, 'snmpgetattack': 6, 'snmpguess': 6,
    'named': 6, 'sendmail': 6, 'xlock': 6, 'xsnoop': 6
}

def preprocess_nslkdd(train_path='data/NSL-KDD/KDDTrain+.txt', 
                     test_path='data/NSL-KDD/KDDTest+.txt'):
    # Preprocess NSL-KDD to common feature space
    print("\n" + "="*70)
    print("PROCESSING NSL-KDD")
    print("="*70)
    
    # Load data
    train_df = pd.read_csv(train_path, names=NSLKDD_COLUMNS)
    test_df = pd.read_csv(test_path, names=NSLKDD_COLUMNS)
    
    print(f"Train samples: {len(train_df):,}")
    print(f"Test samples: {len(test_df):,}")
    
    def extract_common_features(df):
        # Extract common 15 features from NSL-KDD
        features = pd.DataFrame()
        
        # Flow duration
        features['flow_duration'] = df['duration']
        
        # NSL-KDD doesn't have separate fwd/bwd, split by convention
        # Use count as proxy for packets
        features['total_fwd_packets'] = df['count'] / 2
        features['total_bwd_packets'] = df['srv_count'] / 2
        
        # Bytes
        features['src_bytes'] = df['src_bytes']
        features['dst_bytes'] = df['dst_bytes']
        
        # Packet size features - not directly available in NSL-KDD
        # Estimate from bytes / packets
        total_bytes = df['src_bytes'] + df['dst_bytes']
        total_packets = features['total_fwd_packets'] + features['total_bwd_packets'] + 1  # +1 to avoid /0
        features['avg_pkt_size'] = total_bytes / total_packets
        features['min_pkt_size'] = 0  # Not available
        features['max_pkt_size'] = features['avg_pkt_size']  # Approximation
        features['std_pkt_size'] = 0  # Not available
        
        # Flag counts - derive from NSL-KDD 'flag' column
        # NSL-KDD flag values: SF, S0, REJ, RSTR, RSTO, S1, S2, S3, OTH, RSTOS0, SH, RSTRH, SHR
        # Map these to SYN/ACK/FIN/RST indicators
        features['syn_flag_count'] = df['flag'].apply(
            lambda x: 1 if x in ['S0', 'S1', 'S2', 'S3', 'SH'] else 0
        )
        features['ack_flag_count'] = df['flag'].apply(
            lambda x: 1 if x == 'SF' else 0
        )
        features['fin_flag_count'] = df['flag'].apply(
            lambda x: 1 if x == 'SF' else 0
        )
        features['rst_flag_count'] = df['flag'].apply(
            lambda x: 1 if 'RST' in str(x) else 0
        )
        
        # Protocol
        features['protocol_tcp'] = (df['protocol_type'] == 'tcp').astype(int)
        features['protocol_udp'] = (df['protocol_type'] == 'udp').astype(int)
        
        return features[COMMON_FEATURES]
    
    def map_labels(df):
        # Map NSL-KDD labels to unified classes
        labels = df['label'].apply(lambda x: NSLKDD_ATTACK_MAP.get(x.strip().rstrip('.'), -1))
        return labels
    
    X_train = extract_common_features(train_df).values
    X_test = extract_common_features(test_df).values
    y_train = map_labels(train_df).values
    y_test = map_labels(test_df).values
    
    # Filter out unmapped classes (-1)
    train_mask = y_train != -1
    test_mask = y_test != -1
    
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]
    
    print(f"\nAfter filtering: {len(X_train):,} train, {len(X_test):,} test")
    print("\nClass distribution (train):")
    for cls_id, cls_name in UNIFIED_CLASSES.items():
        count = np.sum(y_train == cls_id)
        pct = count / len(y_train) * 100 if len(y_train) > 0 else 0
        print(f"  {cls_name:<20s}: {count:>8,} ({pct:>5.2f}%)")
    
    return X_train, y_train, X_test, y_test


# CICIDS2017 Processing


CICIDS_ATTACK_MAP = {
    # Benign
    'BENIGN': 0,
    # DoS/DDoS
    'DoS Hulk': 1, 'DoS GoldenEye': 1, 'DoS slowloris': 1, 'DoS Slowhttptest': 1,
    'DDoS': 1, 'Heartbleed': 1,
    # Probe
    'PortScan': 2,
    # Brute Force
    'FTP-Patator': 3, 'SSH-Patator': 3,
    # Web Attack
    'Web Attack  Brute Force': 4, 'Web Attack  XSS': 4, 'Web Attack  Sql Injection': 4,
    'Web Attack � Brute Force': 4, 'Web Attack � XSS': 4, 'Web Attack � Sql Injection': 4,
    # Botnet
    'Bot': 5, 'Infiltration': 5
}

def preprocess_cicids2017(path='data/CICIDS2017'):
    # Preprocess CICIDS2017 to common feature space
    print("\n" + "="*70)
    print("PROCESSING CICIDS2017")
    print("="*70)
    
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    print(f"Found {len(files)} files to process")
    
    all_features = []
    all_labels = []
    
    for f in files:
        print(f"\n  Processing: {f}")
        try:
            df = pd.read_csv(os.path.join(path, f), low_memory=False)
            
            # Aggressively strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            # Print columns for debugging on first file
            if f == files[0]:
                print(f"    Columns after stripping (first 10): {list(df.columns[:10])}")
            
            # Handle infinity and NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            if len(df) == 0:
                print(f"    Empty after cleaning, skipping")
                continue
            
            # Check for required columns - use flexible names
            col_mapping = {}
            
            # Find columns by partial matching
            for col in df.columns:
                col_lower = col.lower()
                if 'flow duration' in col_lower:
                    col_mapping['flow_duration'] = col
                elif 'total fwd packets' in col_lower:
                    col_mapping['total_fwd_packets'] = col
                elif 'total backward packets' in col_lower:
                    col_mapping['total_bwd_packets'] = col
                elif 'total length of fwd packets' in col_lower:
                    col_mapping['src_bytes'] = col
                elif 'total length of bwd packets' in col_lower:
                    col_mapping['dst_bytes'] = col
                elif 'average packet size' in col_lower:
                    col_mapping['avg_pkt_size'] = col
                elif 'min packet length' in col_lower:
                    col_mapping['min_pkt_size'] = col
                elif 'max packet length' in col_lower:
                    col_mapping['max_pkt_size'] = col
                elif 'packet length std' in col_lower:
                    col_mapping['std_pkt_size'] = col
                elif 'syn flag count' in col_lower:
                    col_mapping['syn_flag_count'] = col
                elif 'ack flag count' in col_lower:
                    col_mapping['ack_flag_count'] = col
                elif 'fin flag count' in col_lower:
                    col_mapping['fin_flag_count'] = col
                elif 'rst flag count' in col_lower:
                    col_mapping['rst_flag_count'] = col
                elif col_lower == 'protocol':
                    col_mapping['protocol'] = col
                elif col_lower == 'destination port':
                    col_mapping['dest_port'] = col
                elif col_lower == 'label':
                    col_mapping['label'] = col
            
            # Check if we have all required columns (protocol is now optional - can be derived)
            required = ['flow_duration', 'total_fwd_packets', 'total_bwd_packets', 
                       'src_bytes', 'dst_bytes', 'label']
            missing = [r for r in required if r not in col_mapping]
            if missing:
                print(f"     Missing columns: {missing}")
                print(f"    Available columns (sample): {list(df.columns[:20])}")
                continue
            
            # Protocol check - need either protocol column or destination port
            if 'protocol' not in col_mapping and 'dest_port' not in col_mapping:
                print(f"     No Protocol or Destination Port column found")
                continue
            
            # Extract common features using the mapping
            features = pd.DataFrame()
            
            # Flow duration (convert µs to seconds)
            features['flow_duration'] = df[col_mapping['flow_duration']] / 1e6
            
            # Packets
            features['total_fwd_packets'] = df[col_mapping['total_fwd_packets']]
            features['total_bwd_packets'] = df[col_mapping['total_bwd_packets']]
            
            # Bytes
            features['src_bytes'] = df[col_mapping['src_bytes']]
            features['dst_bytes'] = df[col_mapping['dst_bytes']]
            
            # Packet sizes - use defaults if missing
            features['avg_pkt_size'] = df[col_mapping['avg_pkt_size']] if 'avg_pkt_size' in col_mapping else 0
            features['min_pkt_size'] = df[col_mapping['min_pkt_size']] if 'min_pkt_size' in col_mapping else 0
            features['max_pkt_size'] = df[col_mapping['max_pkt_size']] if 'max_pkt_size' in col_mapping else 0
            features['std_pkt_size'] = df[col_mapping['std_pkt_size']] if 'std_pkt_size' in col_mapping else 0
            
            # Flags - use defaults if missing
            features['syn_flag_count'] = df[col_mapping['syn_flag_count']] if 'syn_flag_count' in col_mapping else 0
            features['ack_flag_count'] = df[col_mapping['ack_flag_count']] if 'ack_flag_count' in col_mapping else 0
            features['fin_flag_count'] = df[col_mapping['fin_flag_count']] if 'fin_flag_count' in col_mapping else 0
            features['rst_flag_count'] = df[col_mapping['rst_flag_count']] if 'rst_flag_count' in col_mapping else 0
            
            # Protocol - derive from Protocol column OR Destination Port
            if 'protocol' in col_mapping:
                # CICIDS2017 has numerical protocol (6=TCP, 17=UDP)
                protocol_col = df[col_mapping['protocol']]
                features['protocol_tcp'] = (protocol_col == 6).astype(int)
                features['protocol_udp'] = (protocol_col == 17).astype(int)
            else:
                # Derive protocol from Destination Port (common convention)
                # TCP-typical ports: 80, 443, 21, 22, 23, 25, 110, 143, 993, 995, etc.
                # UDP-typical ports: 53, 67, 68, 69, 123, 161, 162, etc.
                print(f"    Deriving protocol from Destination Port")
                dest_port = df[col_mapping['dest_port']]
                
                # Common UDP ports
                udp_ports = [53, 67, 68, 69, 123, 161, 162, 500, 514, 520, 1900, 5353]
                
                # If there are SYN/ACK flag counts, it's likely TCP
                # Otherwise use port-based heuristic
                if 'syn_flag_count' in col_mapping:
                    syn_col = df[col_mapping['syn_flag_count']]
                    ack_col = df[col_mapping['ack_flag_count']] if 'ack_flag_count' in col_mapping else 0
                    has_tcp_flags = (syn_col > 0) | (ack_col > 0)
                    features['protocol_tcp'] = has_tcp_flags.astype(int)
                    features['protocol_udp'] = (~has_tcp_flags & dest_port.isin(udp_ports)).astype(int)
                else:
                    # Pure port-based
                    features['protocol_udp'] = dest_port.isin(udp_ports).astype(int)
                    features['protocol_tcp'] = (~dest_port.isin(udp_ports)).astype(int)
            
            # Labels
            labels = df[col_mapping['label']].apply(lambda x: CICIDS_ATTACK_MAP.get(str(x).strip(), -1))
            
            all_features.append(features[COMMON_FEATURES])
            all_labels.append(labels)
            
            # Count labels in this file
            valid_labels = labels[labels != -1]
            print(f"    Loaded {len(features):,} samples, {len(valid_labels):,} with valid labels")
            label_counts = Counter(valid_labels)
            if label_counts:
                top_labels = label_counts.most_common(3)
                print(f"    Top labels: {[(UNIFIED_CLASSES[k], v) for k, v in top_labels]}")
        
        except Exception as e:
            print(f"    ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_features:
        raise ValueError("No CICIDS2017 files were processed successfully!")
    
    # Combine all files
    X = pd.concat(all_features, ignore_index=True).values
    y = pd.concat(all_labels, ignore_index=True).values
    
    # Filter out unmapped classes
    mask = y != -1
    X = X[mask]
    y = y[mask]
    
    print(f"\nTotal samples: {len(X):,}")
    print("\nClass distribution:")
    for cls_id, cls_name in UNIFIED_CLASSES.items():
        count = np.sum(y == cls_id)
        pct = count / len(y) * 100 if len(y) > 0 else 0
        print(f"  {cls_name:<20s}: {count:>8,} ({pct:>5.2f}%)")
    
    # Stratified split (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, y_train, X_test, y_test


# CICIoT2023 Processing


def get_ciciot_label_from_filename(filename):
    # Extract label from CICIoT2023 filename
    fname = filename.lower().replace('.pcap.csv', '').replace('.csv', '')
    # Remove trailing digits
    import re
    fname = re.sub(r'\d+$', '', fname)
    
    # Map to unified classes
    if 'benign' in fname:
        return 0  # Benign
    elif 'ddos' in fname or 'dos' in fname:
        return 1  # DoS/DDoS
    elif 'recon' in fname or 'scan' in fname or 'portscan' in fname or 'hostdiscovery' in fname:
        return 2  # Probe
    elif 'brute' in fname or 'dictionary' in fname:
        return 3  # Brute Force
    elif 'xss' in fname or 'sql' in fname or 'command' in fname or 'upload' in fname or 'backdoor' in fname or 'browser' in fname:
        return 4  # Web Attack
    elif 'mirai' in fname:
        return 5  # Botnet/Malware
    elif 'mitm' in fname or 'spoof' in fname:
        return 6  # Spoofing/MITM
    else:
        return -1  # Unknown

def preprocess_ciciot2023(path='data/CICIoT2023', max_samples_per_file=50000):
    """Preprocess CICIoT2023 to common feature space.
    
    Note: CICIoT2023 uses filename for labels!
    """
    print("\n" + "="*70)
    print("PROCESSING CICIoT2023")
    print("="*70)
    
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    print(f"Found {len(files)} files to process")
    print(f"Sampling up to {max_samples_per_file:,} per file to manage memory")
    
    all_features = []
    all_labels = []
    
    for f in files:
        label = get_ciciot_label_from_filename(f)
        if label == -1:
            print(f"  Skipping (unknown label): {f}")
            continue
        
        label_name = UNIFIED_CLASSES[label]
        print(f"\n  Processing: {f} → {label_name}")
        
        try:
            # Read with sampling for memory management
            df = pd.read_csv(os.path.join(path, f), nrows=max_samples_per_file)
            
            # Handle infinity and NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            if len(df) == 0:
                continue
            
            # Extract common features
            features = pd.DataFrame()
            
            # Duration - calculate from rate and number of packets
            # IAT * Number gives total time
            features['flow_duration'] = df['IAT'] * df['Number'] / 1e6  # Convert to seconds
            
            # Packets - CICIoT has combined 'Number' field
            features['total_fwd_packets'] = df['Number'] / 2
            features['total_bwd_packets'] = df['Number'] / 2
            
            # Bytes - use Tot sum split evenly
            features['src_bytes'] = df['Tot sum'] / 2
            features['dst_bytes'] = df['Tot sum'] / 2
            
            # Packet sizes
            features['avg_pkt_size'] = df['AVG']
            features['min_pkt_size'] = df['Min']
            features['max_pkt_size'] = df['Max']
            features['std_pkt_size'] = df['Std']
            
            # Flag counts (use count features)
            features['syn_flag_count'] = df['syn_count']
            features['ack_flag_count'] = df['ack_count']
            features['fin_flag_count'] = df['fin_count']
            features['rst_flag_count'] = df['rst_count']
            
            # Protocol (TCP/UDP are one-hot encoded in CICIoT2023)
            features['protocol_tcp'] = df['TCP']
            features['protocol_udp'] = df['UDP']
            
            # Labels (from filename)
            labels = pd.Series([label] * len(features))
            
            all_features.append(features[COMMON_FEATURES])
            all_labels.append(labels)
            
            print(f"    Loaded {len(features):,} samples")
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    # Combine all files
    X = pd.concat(all_features, ignore_index=True).values
    y = pd.concat(all_labels, ignore_index=True).values
    
    print(f"\nTotal samples: {len(X):,}")
    print("\nClass distribution:")
    for cls_id, cls_name in UNIFIED_CLASSES.items():
        count = np.sum(y == cls_id)
        pct = count / len(y) * 100 if len(y) > 0 else 0
        print(f"  {cls_name:<20s}: {count:>8,} ({pct:>5.2f}%)")
    
    # Stratified split (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, y_train, X_test, y_test


# Main Processing Pipeline


def preprocess_all_datasets(output_dir='data/processed_unified', skip_existing=True):
    """Preprocess all three datasets and save to unified format.
    
    Args:
        output_dir: Where to save processed files
        skip_existing: If True, skip datasets that already have processed files
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("UNIFIED MULTI-DATASET PREPROCESSING")
    print("="*70)
    print(f"\nSkip existing: {skip_existing}")
    print(f"\nCommon Features ({len(COMMON_FEATURES)}):")
    for i, f in enumerate(COMMON_FEATURES):
        print(f"  {i}. {f}")
    
    print(f"\nUnified Classes ({len(UNIFIED_CLASSES)}):")
    for cls_id, cls_name in UNIFIED_CLASSES.items():
        print(f"  {cls_id}. {cls_name}")
    
    results = {}
    
    # Process NSL-KDD
    nslkdd_train_path = f'{output_dir}/nslkdd_X_train.npy'
    if skip_existing and os.path.exists(nslkdd_train_path):
        print("\n⏭️  Skipping NSL-KDD (already processed)")
        X_train = np.load(nslkdd_train_path)
        X_test = np.load(f'{output_dir}/nslkdd_X_test.npy')
        y_train = np.load(f'{output_dir}/nslkdd_y_train.npy')
        y_test = np.load(f'{output_dir}/nslkdd_y_test.npy')
        results['nslkdd'] = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'classes': sorted(list(set(y_train.tolist())))
        }
    elif os.path.exists('data/NSL-KDD'):
        try:
            X_train, y_train, X_test, y_test = preprocess_nslkdd()
            
            # Standardize
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Save
            np.save(f'{output_dir}/nslkdd_X_train.npy', X_train)
            np.save(f'{output_dir}/nslkdd_y_train.npy', y_train)
            np.save(f'{output_dir}/nslkdd_X_test.npy', X_test)
            np.save(f'{output_dir}/nslkdd_y_test.npy', y_test)
            
            results['nslkdd'] = {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'classes': sorted(list(set(y_train.tolist())))
            }
            print(f"\n✓ NSL-KDD saved to {output_dir}/nslkdd_*")
        except Exception as e:
            print(f"\n❌ Error processing NSL-KDD: {e}")
    
    # Process CICIDS2017
    cicids_train_path = f'{output_dir}/cicids2017_X_train.npy'
    if skip_existing and os.path.exists(cicids_train_path):
        print("\n⏭️  Skipping CICIDS2017 (already processed)")
        X_train = np.load(cicids_train_path)
        X_test = np.load(f'{output_dir}/cicids2017_X_test.npy')
        y_train = np.load(f'{output_dir}/cicids2017_y_train.npy')
        y_test = np.load(f'{output_dir}/cicids2017_y_test.npy')
        results['cicids2017'] = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'classes': sorted(list(set(y_train.tolist())))
        }
    elif os.path.exists('data/CICIDS2017'):
        try:
            X_train, y_train, X_test, y_test = preprocess_cicids2017()
            
            # Standardize
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Save
            np.save(f'{output_dir}/cicids2017_X_train.npy', X_train)
            np.save(f'{output_dir}/cicids2017_y_train.npy', y_train)
            np.save(f'{output_dir}/cicids2017_X_test.npy', X_test)
            np.save(f'{output_dir}/cicids2017_y_test.npy', y_test)
            
            results['cicids2017'] = {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'classes': sorted(list(set(y_train.tolist())))
            }
            print(f"\n✓ CICIDS2017 saved to {output_dir}/cicids2017_*")
        except Exception as e:
            print(f"\n❌ Error processing CICIDS2017: {e}")
            import traceback
            traceback.print_exc()
    
    # Process CICIoT2023
    ciciot_train_path = f'{output_dir}/ciciot2023_X_train.npy'
    if skip_existing and os.path.exists(ciciot_train_path):
        print("\n⏭️  Skipping CICIoT2023 (already processed)")
        X_train = np.load(ciciot_train_path)
        X_test = np.load(f'{output_dir}/ciciot2023_X_test.npy')
        y_train = np.load(f'{output_dir}/ciciot2023_y_train.npy')
        y_test = np.load(f'{output_dir}/ciciot2023_y_test.npy')
        results['ciciot2023'] = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'classes': sorted(list(set(y_train.tolist())))
        }
    elif os.path.exists('data/CICIoT2023'):
        try:
            X_train, y_train, X_test, y_test = preprocess_ciciot2023()
            
            # Standardize
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Save
            np.save(f'{output_dir}/ciciot2023_X_train.npy', X_train)
            np.save(f'{output_dir}/ciciot2023_y_train.npy', y_train)
            np.save(f'{output_dir}/ciciot2023_X_test.npy', X_test)
            np.save(f'{output_dir}/ciciot2023_y_test.npy', y_test)
            
            results['ciciot2023'] = {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'classes': sorted(list(set(y_train.tolist())))
            }
            print(f"\n✓ CICIoT2023 saved to {output_dir}/ciciot2023_*")
        except Exception as e:
            print(f"\n❌ Error processing CICIoT2023: {e}")
    
    # Save metadata
    metadata = {
        'common_features': COMMON_FEATURES,
        'unified_classes': {str(k): v for k, v in UNIFIED_CLASSES.items()},
        'datasets': results,
        'num_features': len(COMMON_FEATURES),
        'num_classes': len(UNIFIED_CLASSES)
    }
    
    with open(f'{output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\nSaved to: {output_dir}/")
    print("\nSummary:")
    for dataset, info in results.items():
        print(f"\n{dataset.upper()}:")
        print(f"  Train: {info['train_samples']:,}")
        print(f"  Test: {info['test_samples']:,}")
        print(f"  Classes present: {info['classes']}")
    
    print(f"\n✓ Metadata saved to: {output_dir}/metadata.json")
    print("\nNext step: Run federated multi-dataset continual learning!")

if __name__ == '__main__':
    preprocess_all_datasets()