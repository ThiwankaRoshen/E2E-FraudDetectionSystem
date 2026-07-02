import json, logging, os, time, subprocess
from typing import Dict, List, Any, Optional
from confluent_kafka import Producer, Consumer, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic, ConfigResource
from datetime import datetime
import pandas as pd
 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NativeKafkaConfig:
    """Configuration manager for native Kafka setup"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize native Kafka configuration
        
        Args:
            config_path: Path to configuration file (currently not used, loads default config.yaml)
        """ 
        # Read bootstrap servers from environment or default to localhost
        self.bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        
    def get_producer_config(self) -> Dict[str, Any]:
        """Get producer configuration for native Kafka"""
        return {
                'bootstrap.servers': self.bootstrap_servers,
                'acks': '1',
                'retries': 1,
                'enable.idempotence': False,
                'compression.type': 'none',
                'batch.size': 1024,
                'linger.ms': 0
                # Simplified config to fix connection issues
                }
        
    def get_consumer_config(self, group_id: str) -> Dict[str, Any]:
        """Get consumer configuration for native Kafka"""
        return {
                'bootstrap.servers': self.bootstrap_servers,
                'group.id': group_id,
                'auto.offset.reset': 'earliest',
                'enable.auto.commit': True,
                'auto.commit.interval.ms': 1000,
                'session.timeout.ms': 30000,
                'max.poll.records': 500
                }


class NativeKafkaValidator:
    """Validator for native Kafka installation and connectivity"""
    
    @staticmethod
    def check_kafka_installation() -> Dict[str, Any]:
        """
        Check if Kafka is properly installed natively
        
        Returns:
            Dict with installation status
        """
        result = {
            'kafka_home_set': False,
            'kafka_binaries_found': False,
            'java_available': False,
            'error_messages': []
        }
        
        try:
            # Check KAFKA_HOME environment variable
            kafka_home = os.environ.get('KAFKA_HOME')
            if kafka_home:
                result['kafka_home_set'] = True
                logger.info(f"KAFKA_HOME found: {kafka_home}")
            else:
                # Try to detect Kafka installation automatically
                common_kafka_paths = [
                    '/opt/homebrew/opt/kafka/libexec',  # Homebrew on Apple Silicon
                    '/usr/local/opt/kafka/libexec',    # Homebrew on Intel
                    '/opt/kafka',                       # Common Linux path
                    '/usr/local/kafka',                 # Alternative Linux path
                ]
                
                for path in common_kafka_paths:
                    if os.path.exists(os.path.join(path, 'bin', 'kafka-topics.sh')):
                        result['kafka_home_set'] = True
                        logger.info(f"Kafka installation detected at: {path}")
                        # Set KAFKA_HOME for this session
                        os.environ['KAFKA_HOME'] = path
                        # Add Kafka bin to PATH for this session
                        kafka_bin_path = os.path.join(path, 'bin')
                        current_path = os.environ.get('PATH', '')
                        if kafka_bin_path not in current_path:
                            os.environ['PATH'] = f"{kafka_bin_path}:{current_path}"
                            logger.info(f"Added Kafka bin to PATH: {kafka_bin_path}")
                        break
                
                if not result['kafka_home_set']:
                    result['error_messages'].append("KAFKA_HOME environment variable not set and Kafka not found in common paths")
                    logger.warning("❌ Kafka installation not detected")
            
            # Check if kafka-topics.sh is available
            try:
                subprocess.run(['kafka-topics.sh', '--version'], 
                             capture_output=True, check=True, timeout=10)
                result['kafka_binaries_found'] = True
                logger.info("Kafka binaries found in PATH")
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                result['error_messages'].append("Kafka binaries not found in PATH")
            
            # Check Java installation
            try:
                java_result = subprocess.run(['java', '-version'], 
                                           capture_output=True, check=True, timeout=10)
                result['java_available'] = True
                logger.info("Java runtime available")
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                result['error_messages'].append("Java runtime not found")
            
        except Exception as e:
            result['error_messages'].append(f"Installation check failed: {str(e)}")
        
        return result
    
    @staticmethod
    def check_kafka_connection(bootstrap_servers: str = None) -> bool:
        """
        Check if native Kafka broker is running and accessible
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            
        Returns:
            True if connection successful, False otherwise
        """
        if bootstrap_servers is None:
            bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        try:
            admin_client = AdminClient({'bootstrap.servers': bootstrap_servers})
            metadata = admin_client.list_topics(timeout=5)
            logger.info(f"Successfully connected to native Kafka broker at {bootstrap_servers}")
            return True
            
        except Exception as e:
            logger.error(f"Cannot connect to native Kafka broker at {bootstrap_servers}: {str(e)}")
            return False


def create_topic(topic_name: str, partitions: int = 1, replication_factor: int = 1) -> bool:
    """
    Create Kafka topic on native broker
    
    Args:
        topic_name: Name of the topic to create
        partitions: Number of partitions
        replication_factor: Replication factor
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Teaching Note: Using confluent-kafka admin client for native broker operations
        bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        admin_client = AdminClient({'bootstrap.servers': bootstrap_servers})
        
        # Create topic
        topic_list = [NewTopic(
            topic=topic_name,
            num_partitions=partitions,
            replication_factor=replication_factor
        )]
        
        # Execute creation
        fs = admin_client.create_topics(topic_list)
        
        # Wait for result
        for topic, f in fs.items():
            try:
                f.result()  # The result itself is None
                logger.info(f"Topic '{topic}' created successfully on native broker")
                return True
            except Exception as e:
                if "TopicExistsException" in str(e):
                    logger.info(f"Topic '{topic}' already exists")
                    return True
                else:
                    logger.error(f"Failed to create topic '{topic}': {e}")
                    return False
                    
    except Exception as e:
        logger.error(f"Error creating topic '{topic_name}': {str(e)}")
        return False




def check_kafka_connection() -> bool:
    """
    Check if native Kafka broker is running and accessible
    
    Returns:
        True if connection successful, False otherwise
    """
    return NativeKafkaValidator.check_kafka_connection()


def validate_native_setup() -> Dict[str, Any]:
    """
    Comprehensive validation of native Kafka setup
    
    Returns:
        Dict with validation results
    """
    logger.info("Validating native Kafka setup...")
    
    # Check installation
    installation_status = NativeKafkaValidator.check_kafka_installation()
    
    # Check broker connection
    broker_running = NativeKafkaValidator.check_kafka_connection()
    
    # Overall status - focus on what's essential for functionality
    setup_valid = (
        installation_status['kafka_binaries_found'] and
        installation_status['java_available']
        # Note: broker_running is checked separately as it requires the broker to be started
    )
    
    validation_result = {
        'setup_valid': setup_valid,
        'installation_status': installation_status,
        'broker_running': broker_running,
        'recommendations': []
    }
    
    # Add recommendations based on issues found
    if not installation_status['kafka_home_set']:
        validation_result['recommendations'].append(
            "Set KAFKA_HOME environment variable to your Kafka installation directory"
        )
    
    if not installation_status['kafka_binaries_found']:
        validation_result['recommendations'].append(
            "Install Apache Kafka natively using: brew install kafka (macOS) or download from https://kafka.apache.org/downloads"
        )
    
    if not installation_status['java_available']:
        validation_result['recommendations'].append(
            "Install Java 17 or higher (required for Kafka)"
        )
    
    if not broker_running:
        validation_result['recommendations'].append(
            "Start native Kafka broker using: make kafka-start"
        )
    
    if setup_valid:
        if broker_running:
            logger.info("✅ Native Kafka setup is valid and broker is running")
        else:
            logger.info("✅ Native Kafka setup is valid (broker not running - this is normal)")
            logger.info("💡 To start broker: make kafka-format && make kafka-start")
    else:
        logger.warning("⚠️ Native Kafka setup has issues")
        for rec in validation_result['recommendations']:
            logger.warning(f"  - {rec}")
    
    return validation_result


class NativeKafkaProducer:
    """Enhanced producer for native Kafka broker"""
    
    def __init__(self):
        """Initialize native Kafka producer"""
        self.config = NativeKafkaConfig()
        self.producer = None
        self._connect()
    
    def _connect(self):
        """Connect to native Kafka broker"""
        try:
            # Validate setup first
            if not check_kafka_connection():
                raise ConnectionError("Cannot connect to native Kafka broker at localhost:9092")
            
            self.producer = Producer(self.config.get_producer_config())
            logger.info("Connected to native Kafka broker")
            
        except Exception as e:
            logger.error(f"Error connecting to native Kafka broker: {str(e)}")
            raise
    
    def send_message(self, topic: str, message: Dict[str, Any], key: str = None) -> bool:
        """
        Send message to native Kafka broker
        
        Args:
            topic: Topic name
            message: Message data
            key: Optional message key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Enrich message with metadata
            enriched_message = {
                'timestamp': datetime.utcnow().isoformat(),
                'data': message
            }
            
            # Send to native broker
            message_json = json.dumps(enriched_message, default=str)
            self.producer.produce(
                topic=topic,
                key=key,
                value=message_json,
                callback=self._delivery_callback
            )
            
            # Fast delivery - just poll once and return
            self.producer.poll(0)
            return True
            
        except Exception as e:
            logger.error(f"Error sending message to native broker: {str(e)}")
            return False
    
    def _delivery_callback(self, err, msg):
        """Delivery report callback"""
        if err is not None:
            logger.error(f'Message delivery failed: {err}')
        else:
            logger.debug(f'Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}')
    
    def close(self):
        """Close producer connection"""
        if self.producer:
            self.producer.flush()
            logger.info("Native Kafka producer closed")


class NativeKafkaConsumer:
    """Enhanced consumer for native Kafka broker"""
    
    def __init__(self, group_id: str, topics: List[str]):
        """
        Initialize native Kafka consumer
        
        Args:
            group_id: Consumer group ID
            topics: List of topics to subscribe to
        """
        self.config = NativeKafkaConfig()
        self.group_id = group_id
        self.topics = topics
        self.consumer = None
        self._connect()
    
    def _connect(self):
        """Connect to native Kafka broker"""
        try:
            # Validate setup first
            if not check_kafka_connection():
                raise ConnectionError("Cannot connect to native Kafka broker at localhost:9092")
            
            self.consumer = Consumer(self.config.get_consumer_config(self.group_id))
            self.consumer.subscribe(self.topics)
            logger.info(f"Connected to native Kafka broker, subscribed to: {self.topics}")
            
        except Exception as e:
            logger.error(f"Error connecting to native Kafka broker: {str(e)}")
            raise
    
    def consume_to_dataframe(self, max_messages: int = 1000, timeout: int = 30) -> pd.DataFrame:
        """
        Consume messages and return as DataFrame
        
        Args:
            max_messages: Maximum number of messages to consume
            timeout: Timeout in seconds
            
        Returns:
            DataFrame with consumed messages
        """
        messages = []
        count = 0
        start_time = time.time()
        
        try:
            while count < max_messages and (time.time() - start_time) < timeout:
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    logger.error(f"Consumer error: {msg.error()}")
                    continue
                
                try:
                    message_data = json.loads(msg.value().decode('utf-8'))
                    data = message_data.get('data', {})
                    
                    # Add Kafka metadata
                    data['_kafka_topic'] = msg.topic()
                    data['_kafka_partition'] = msg.partition()
                    data['_kafka_offset'] = msg.offset()
                    data['_kafka_timestamp'] = message_data.get('timestamp')
                    
                    messages.append(data)
                    count += 1
                    
                except json.JSONDecodeError:
                    logger.warning("Failed to decode message as JSON")
                    continue
            
            return pd.DataFrame(messages)
            
        except Exception as e:
            logger.error(f"Error consuming messages: {str(e)}")
            return pd.DataFrame()
    
    def close(self):
        """Close consumer connection"""
        if self.consumer:
            self.consumer.close()
            logger.info("Native Kafka consumer closed")

