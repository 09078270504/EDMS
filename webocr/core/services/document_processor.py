import os
import re
import time
import logging
import concurrent.futures
from pathlib import Path
from multiprocessing import cpu_count
from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils import timezone
import pytz
import threading
import gc
import torch
from core.services.ocr_processor import OCRProcessor
from core.services.qwen_extractor import QwenExtractor
from core.services.archive_manager import ArchiveManager

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Document processor that adapts to different document types"""

    def __init__(self):
        # Simplified worker configuration
        self.max_workers = min(cpu_count(), 4)
        
        # Model instances with thread safety
        self._ocr_processor = None
        self._data_extractor = None
        self._model_lock = threading.Lock()
        
        self.archive_manager = ArchiveManager()
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'total_time': 0,
            'document_types': {}
        }
        
        # Compile patterns for better performance
        self._compile_patterns()
        
        logger.info(f"Initialized Flexible Document Processor with {self.max_workers} workers")
        self._initialize_models()
    
    def _compile_patterns(self):
        """Compile regex patterns for better performance"""
        # Person name patterns - more comprehensive and accurate
        self.person_patterns = [
            # Full names with titles
            re.compile(r'\b(?:Mr|Ms|Mrs|Dr|Prof|Sir|Madam|Miss|Atty|Engr|Archt)\.?\s+([A-Z][a-z]{1,15}(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]{1,15}(?:\s+(?:Jr|Sr|III?|IV)\.?)?)\b'),
            
            # Full names without titles (First Middle Last, First Last)
            re.compile(r'\b([A-Z][a-z]{2,15}\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]{2,15}(?:\s+(?:Jr|Sr|III?|IV)\.?)?)\b(?!\s+(?:Inc|Corp|LLC|Ltd|Co|Corporation|Company|Technologies|Solutions|Systems|Services|University|College|School|Hospital|Clinic|Bank|Group|Holdings|International|Enterprises|Foundation|Institute|Organization|Department|Ministry|Office|Bureau|Agency|Commission|Authority|Council)\.?)'),
            
            # Names with middle initials
            re.compile(r'\b([A-Z][a-z]{2,15}\s+[A-Z]\.\s+[A-Z][a-z]{2,15}(?:\s+(?:Jr|Sr|III?|IV)\.?)?)\b'),
            
            # Filipino names patterns
            re.compile(r'\b([A-Z][a-z]{2,15}(?:\s+(?:de|del|delos|dela|de\s+la)\s+)?[A-Z][a-z]{2,15}(?:\s+(?:Jr|Sr|III?|IV)\.?)?)\b(?!\s+(?:Inc|Corp|LLC|Ltd|Co|Corporation|Company|Technologies|Solutions|Systems|Services))'),
        ]
        
        # Company patterns - more comprehensive
        self.company_patterns = [
            # Standard business entities
            re.compile(r'\b([A-Z][A-Za-z\s&\-\'\.]{2,50}(?:\s+(?:Inc|Corp|LLC|Ltd|Co|Corporation|Company|Technologies|Solutions|Systems|Services|International|Enterprises|Holdings|Group|Partners|Associates|Consultants|Contractors|Suppliers|Manufacturers|Industries|Trading|Investment|Development|Management|Foundation|Institute|Organization|University|College|School|Hospital|Clinic|Bank|Insurance|Finance|Securities|Properties|Realty|Construction|Engineering|Design|Marketing|Advertising|Communications|Media|Publishing|Printing|Transportation|Logistics|Shipping|Aviation|Maritime|Energy|Power|Oil|Gas|Mining|Agriculture|Pharmaceuticals|Healthcare|Medical|Dental|Legal|Accounting|Audit|Tax|Consulting|Advisory|Training|Education|Research|Laboratory|Technology|Software|Hardware|Electronics|Telecommunications|Broadcasting|Entertainment|Gaming|Sports|Travel|Tourism|Hospitality|Restaurant|Food|Beverage|Retail|Wholesale|Distribution|Import|Export|General\s+Services)\.?))\b'),
            
            # Government agencies and institutions
            re.compile(r'\b((?:Department|Ministry|Office|Bureau|Agency|Commission|Authority|Council)\s+of\s+[A-Z][A-Za-z\s&\-\']{5,50})\b'),
            
            # Acronyms (3-6 letters, all caps)
            re.compile(r'\b([A-Z]{3,6}(?:\s+[A-Z]{2,4})*)\b(?!\s*[a-z])'),
            
            # Banks and financial institutions
            re.compile(r'\b([A-Z][A-Za-z\s&\-\'\.]{2,40}\s+(?:Bank|Banking|Finance|Financial|Investment|Securities|Insurance|Credit\s+Union|Cooperative|Trust|Fund|Capital|Asset\s+Management|Wealth\s+Management))\b'),
            
            # Educational institutions
            re.compile(r'\b([A-Z][A-Za-z\s&\-\'\.]{2,40}\s+(?:University|College|School|Institute|Academy|Seminary|Polytechnic|Technical\s+College|Community\s+College))\b'),
        ]
        
        # Location patterns - more comprehensive
        self.location_patterns = [
            # Cities with common location indicators
            re.compile(r'\b([A-Z][a-z]{2,25}(?:\s+[A-Z][a-z]{2,25})*)\s+(?:City|Province|State|County|Municipality|District|Region|Area|Zone|Subdivision|Village|Town|Barangay|Metro|NCR|CAR)\b'),
            
            # Business locations
            re.compile(r'\b([A-Z][a-z]{2,25}(?:\s+[A-Z][a-z]{2,25})*)\s+(?:Plant|Branch|Office|Warehouse|Department|Division|Facility|Complex|Center|Centre|Mall|Plaza|Building|Tower|Hub|Terminal|Port|Airport|Station)\b'),
            
            # Address components
            re.compile(r'\b([A-Z][a-z]{2,25}(?:\s+[A-Z][a-z]{2,25})*)\s+(?:Street|Road|Avenue|Boulevard|Drive|Lane|Court|Circle|Plaza|Square|Subdivision|Village|Heights|Hills|Gardens|Park|Residences)\b'),
            
            # International locations
            re.compile(r'\b([A-Z][a-z]{2,25}(?:\s+[A-Z][a-z]{2,25})*),?\s+(?:Philippines|USA|United\s+States|Singapore|Malaysia|Thailand|Indonesia|Vietnam|Japan|Korea|China|Hong\s+Kong|Taiwan|Australia|Canada|UK|United\s+Kingdom|Germany|France|Italy|Spain|Netherlands|Belgium|Switzerland|Dubai|UAE|Qatar|Saudi\s+Arabia|India|Pakistan|Bangladesh|Sri\s+Lanka|Myanmar|Cambodia|Laos|Brunei)\b'),
            
            # Philippine specific locations
            re.compile(r'\b(Metro\s+Manila|Makati|Taguig|BGC|Ortigas|Alabang|Quezon\s+City|Manila|Pasig|Mandaluyong|San\s+Juan|Marikina|Pasay|Paranaque|Las\s+Pinas|Muntinlupa|Pateros|Valenzuela|Malabon|Navotas|Caloocan|Cebu|Davao|Iloilo|Bacolod|Cagayan\s+de\s+Oro|Zamboanga|General\s+Santos|Baguio|Angeles|Clark|Subic|Batangas|Laguna|Cavite|Rizal|Bulacan|Pampanga|Tarlac|Nueva\s+Ecija|Pangasinan|La\s+Union|Ilocos\s+Norte|Ilocos\s+Sur|Cagayan|Isabela|Quirino|Nueva\s+Vizcaya|Ifugao|Mountain\s+Province|Kalinga|Apayao|Abra|Benguet)\b'),
        ]
        
        # Enhanced date patterns
        self.date_patterns = [
            # ISO format
            re.compile(r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b'),
            
            # US format
            re.compile(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b'),
            
            # European format  
            re.compile(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b'),
            
            # Month names - full and abbreviated
            re.compile(r'\b(\d{1,2}[\s\-/]*(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s\-/]*\d{4})\b', re.IGNORECASE),
            
            # Reverse month name format
            re.compile(r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s\-/]*\d{1,2}[\s\-/,]*\d{4})\b', re.IGNORECASE),
            
            # Day-Month-Year with names
            re.compile(r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s,]+\d{4})\b', re.IGNORECASE),
            
            # Relative dates
            re.compile(r'\b((?:today|yesterday|tomorrow|last\s+(?:week|month|year)|next\s+(?:week|month|year)|this\s+(?:week|month|year)))\b', re.IGNORECASE),
            
            # Quarter references
            re.compile(r'\b(Q[1-4]\s+\d{4}|(?:first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+\d{4})\b', re.IGNORECASE),
        ]
        
        # Enhanced amount patterns
        self.amount_patterns = [
            # Currency with amount
            re.compile(r'(?:PHP|USD|EUR|GBP|JPY|SGD|MYR|THB|IDR|VND|KRW|CNY|HKD|AUD|CAD|CHF|SEK|NOK|DKK|PLN|CZK|HUF|ZAR|BRL|MXN|ARS|CLP|COP|PEN|UYU|BOB|PYG|INR|PKR|BDT|LKR|NPR|BTN|MVR|AFN|MMK|KHR|LAK|BND)\s*[\$¥€£₹₩₽₨₪₦₡₵₴₸₼₾]?\s*(\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,4})?)', re.IGNORECASE),
            
            # Amount with currency after
            re.compile(r'\b(\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,4})?)\s*(?:PHP|USD|EUR|GBP|JPY|SGD|MYR|THB|IDR|VND|KRW|CNY|HKD|AUD|CAD|CHF|SEK|NOK|DKK|PLN|CZK|HUF|ZAR|BRL|MXN|ARS|CLP|COP|PEN|UYU|BOB|PYG|INR|PKR|BDT|LKR|NPR|BTN|MVR|AFN|MMK|KHR|LAK|BND)\b', re.IGNORECASE),
            
            # Currency symbols
            re.compile(r'[\$¥€£₹₩₽₨₪₦₡₵₴₸₼₾]\s*(\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,4})?)'),
            
            # Amount with words
            re.compile(r'\b(\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,4})?)\s*(?:pesos|dollars|euros|pounds|yen|ringgit|baht|rupiah|dong|won|yuan|cents|million|billion|thousand|k|m|b)\b', re.IGNORECASE),
            
            # Written amounts
            re.compile(r'\b((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\s+(?:hundred|thousand|million|billion)\s*(?:pesos|dollars|euros|pounds|yen)?)\b', re.IGNORECASE),
        ]
        
        # Common false positive patterns to exclude
        self.false_positive_patterns = {
            'people': [
                re.compile(r'\b(?:Date|Time|Page|Total|Amount|Number|Code|ID|Reference|Subject|Address|Phone|Email|Website|Department|Section|Unit|Room|Floor|Building|Street|Road|Avenue|City|Province|Country|Year|Month|Day|Week|Quarter|Report|Document|File|Attachment|Version|Copy|Original|Draft|Final|Approved|Pending|Completed|Cancelled|Active|Inactive|Valid|Invalid|Public|Private|Confidential|Urgent|Important|High|Medium|Low|New|Old|Recent|Current|Previous|Next|First|Last|Primary|Secondary|Main|Sub|General|Specific|Special|Regular|Standard|Custom|Default|Optional|Required|Mandatory|Automatic|Manual|Online|Offline|Digital|Physical|Virtual|Actual|Estimated|Projected|Planned|Scheduled|Expected|Confirmed|Tentative|Preliminary|Temporary|Permanent|Short|Long|Quick|Slow|Fast|Easy|Difficult|Simple|Complex|Basic|Advanced|Professional|Personal|Business|Commercial|Industrial|Residential|Educational|Medical|Legal|Financial|Technical|Administrative|Operational|Strategic|Tactical|Local|National|International|Global|Regional|Domestic|Foreign|Internal|External|Incoming|Outgoing|Received|Sent|Delivered|Returned|Forwarded|Copied|Moved|Deleted|Saved|Updated|Modified|Changed|Created|Generated|Produced|Manufactured|Processed|Handled|Managed|Supervised|Controlled|Monitored|Tracked|Recorded|Logged|Reported|Analyzed|Reviewed|Approved|Rejected|Accepted|Declined|Confirmed|Cancelled|Postponed|Rescheduled|Extended|Reduced|Increased|Decreased|Improved|Enhanced|Upgraded|Downgraded|Replaced|Renewed|Expired|Terminated|Suspended|Resumed|Started|Stopped|Paused|Continued|Finished|Completed|Closed|Opened|Locked|Unlocked|Secured|Protected|Encrypted|Decrypted|Compressed|Decompressed|Archived|Restored|Backed|Synchronized|Uploaded|Downloaded|Imported|Exported|Converted|Transformed|Translated|Formatted|Printed|Scanned|Copied|Pasted|Cut|Deleted|Inserted|Added|Removed|Cleared|Reset|Refreshed|Reloaded|Restarted|Shutdown|Powered|Connected|Disconnected|Linked|Unlinked|Attached|Detached|Mounted|Unmounted|Installed|Uninstalled|Configured|Reconfigured|Calibrated|Tested|Validated|Verified|Authenticated|Authorized|Logged|Registered|Unregistered|Subscribed|Unsubscribed|Enrolled|Disenrolled|Activated|Deactivated|Enabled|Disabled|Allowed|Blocked|Granted|Denied|Permitted|Prohibited|Restricted|Unrestricted|Limited|Unlimited|Available|Unavailable|Accessible|Inaccessible|Visible|Hidden|Shown|Displayed|Presented|Demonstrated|Explained|Described|Detailed|Summarized|Listed|Indexed|Categorized|Classified|Grouped|Sorted|Filtered|Searched|Found|Located|Identified|Recognized|Detected|Discovered|Explored|Investigated|Examined|Inspected|Evaluated|Assessed|Measured|Calculated|Computed|Estimated|Approximated|Rounded|Truncated|Expanded|Collapsed|Minimized|Maximized|Optimized|Customized|Personalized|Standardized|Normalized|Regularized|Stabilized|Balanced|Aligned|Centered|Positioned|Located|Placed|Arranged|Organized|Structured|Designed|Planned|Prepared|Developed|Built|Constructed|Created|Generated|Produced|Manufactured|Assembled|Installed|Deployed|Implemented|Executed|Performed|Conducted|Operated|Managed|Administered|Supervised|Controlled|Directed|Guided|Led|Coordinated|Organized|Arranged|Scheduled|Planned|Designed|Developed|Created|Built|Constructed|Established|Founded|Started|Initiated|Launched|Introduced|Presented|Delivered|Provided|Supplied|Offered|Available|Ready|Prepared|Completed|Finished|Done|Accomplished|Achieved|Attained|Reached|Obtained|Acquired|Gained|Earned|Won|Lost|Failed|Succeeded|Passed|Completed|Graduated|Certified|Licensed|Registered|Qualified|Eligible|Authorized|Approved|Accepted|Admitted|Enrolled|Subscribed|Joined|Participated|Attended|Visited|Toured|Traveled|Moved|Relocated|Transferred|Shifted|Changed|Switched|Converted|Transformed|Adapted|Adjusted|Modified|Altered|Updated|Upgraded|Improved|Enhanced|Refined|Polished|Perfected|Completed|Finalized|Concluded|Ended|Terminated|Stopped|Ceased|Discontinued|Suspended|Paused|Interrupted|Delayed|Postponed|Rescheduled|Extended|Prolonged|Shortened|Reduced|Decreased|Minimized|Limited|Restricted|Constrained|Bounded|Contained|Enclosed|Surrounded|Protected|Secured|Guarded|Defended|Shielded|Covered|Hidden|Concealed|Masked|Disguised|Camouflaged|Obscured|Blocked|Prevented|Avoided|Evaded|Escaped|Fled|Retreated|Withdrawn|Removed|Eliminated|Deleted|Erased|Cleared|Cleaned|Purged|Flushed|Emptied|Vacated|Abandoned|Deserted|Left|Departed|Exited|Quit|Resigned|Retired|Stepped|Backed|Returned|Came|Arrived|Entered|Joined|Participated|Engaged|Involved|Included|Incorporated|Integrated|Combined|Merged|United|Connected|Linked|Attached|Bound|Tied|Secured|Fastened|Fixed|Stable|Steady|Consistent|Reliable|Dependable|Trustworthy|Credible|Believable|Convincing|Persuasive|Compelling|Attractive|Appealing|Interesting|Engaging|Entertaining|Enjoyable|Pleasant|Comfortable|Convenient|Easy|Simple|Straightforward|Clear|Obvious|Evident|Apparent|Visible|Noticeable|Remarkable|Notable|Significant|Important|Major|Minor|Small|Large|Big|Huge|Enormous|Massive|Tiny|Minute|Microscopic|Gigantic|Colossal|Tremendous|Immense|Vast|Extensive|Comprehensive|Complete|Full|Partial|Incomplete|Total|Whole|Entire|All|Every|Each|Some|Few|Many|Most|Several|Various|Different|Similar|Same|Identical|Equal|Equivalent|Comparable|Relative|Absolute|Exact|Precise|Accurate|Correct|Right|Wrong|Incorrect|False|True|Valid|Invalid|Legal|Illegal|Legitimate|Illegitimate|Authorized|Unauthorized|Official|Unofficial|Formal|Informal|Professional|Amateur|Expert|Novice|Beginner|Advanced|Experienced|Skilled|Talented|Gifted|Capable|Competent|Qualified|Certified|Licensed|Registered|Accredited|Approved|Endorsed|Recommended|Suggested|Proposed|Offered|Available|Accessible|Obtainable|Achievable|Possible|Probable|Likely|Unlikely|Impossible|Improbable|Certain|Uncertain|Sure|Unsure|Confident|Doubtful|Positive|Negative|Optimistic|Pessimistic|Hopeful|Hopeless|Encouraged|Discouraged|Motivated|Demotivated|Inspired|Uninspired|Enthusiastic|Unenthusiastic|Excited|Bored|Interested|Uninterested|Curious|Indifferent|Concerned|Unconcerned|Worried|Relaxed|Stressed|Calm|Anxious|Peaceful|Troubled|Disturbed|Upset|Happy|Sad|Joyful|Sorrowful|Cheerful|Gloomy|Bright|Dark|Light|Heavy|Strong|Weak|Powerful|Powerless|Effective|Ineffective|Efficient|Inefficient|Productive|Unproductive|Successful|Unsuccessful|Profitable|Unprofitable|Beneficial|Harmful|Useful|Useless|Valuable|Worthless|Important|Unimportant|Necessary|Unnecessary|Essential|Nonessential|Critical|Noncritical|Urgent|Nonurgent|Priority|Routine|Special|Normal|Regular|Irregular|Standard|Nonstandard|Typical|Atypical|Common|Uncommon|Usual|Unusual|Ordinary|Extraordinary|Average|Exceptional|Outstanding|Mediocre|Superior|Inferior|Better|Worse|Best|Worst|Good|Bad|Excellent|Poor|Great|Terrible|Wonderful|Awful|Amazing|Disappointing|Impressive|Unimpressive|Remarkable|Unremarkable|Notable|Forgettable|Memorable|Unforgettable|Significant|Insignificant|Meaningful|Meaningless|Relevant|Irrelevant|Applicable|Inapplicable|Suitable|Unsuitable|Appropriate|Inappropriate|Proper|Improper|Correct|Incorrect|Right|Wrong|Accurate|Inaccurate|Precise|Imprecise|Exact|Inexact|Specific|General|Detailed|Vague|Clear|Unclear|Obvious|Obscure|Evident|Hidden|Apparent|Concealed|Visible|Invisible|Noticeable|Unnoticeable|Recognizable|Unrecognizable|Familiar|Unfamiliar|Known|Unknown|Identified|Unidentified|Named|Unnamed|Labeled|Unlabeled|Tagged|Untagged|Marked|Unmarked|Signed|Unsigned|Sealed|Unsealed|Locked|Unlocked|Secured|Unsecured|Protected|Unprotected|Safe|Unsafe|Dangerous|Harmless|Risky|Risk-free|Hazardous|Safe|Threatening|Nonthreatening|Warning|Reassuring|Alarming|Calming|Exciting|Boring|Thrilling|Dull|Interesting|Uninteresting|Engaging|Unengaging|Captivating|Repelling|Attractive|Unattractive|Beautiful|Ugly|Pretty|Plain|Handsome|Homely|Gorgeous|Hideous|Stunning|Shocking|Impressive|Unimpressive|Spectacular|Ordinary|Magnificent|Mediocre|Splendid|Dreadful|Superb|Awful|Excellent|Terrible|Outstanding|Poor|Exceptional|Average|Extraordinary|Normal|Special|Regular|Unique|Common|Rare|Frequent|Infrequent|Often|Seldom|Always|Never|Sometimes|Usually|Occasionally|Regularly|Irregularly|Constantly|Intermittently|Continuously|Discontinuously|Permanently|Temporarily|Forever|Briefly|Long|Short|Extended|Brief|Prolonged|Quick|Slow|Fast|Rapid|Gradual|Sudden|Immediate|Delayed|Instant|Eventual|Prompt|Late|Early|Timely|Untimely|Scheduled|Unscheduled|Planned|Unplanned|Expected|Unexpected|Predictable|Unpredictable|Anticipated|Unanticipated|Foreseen|Unforeseen|Prepared|Unprepared|Ready|Unready|Available|Unavailable|Present|Absent|Here|There|Nearby|Distant|Close|Far|Near|Remote|Local|Foreign|Domestic|International|National|Regional|Global|Worldwide|Universal|Limited|Unlimited|Restricted|Unrestricted|Bounded|Unbounded|Finite|Infinite|Measurable|Immeasurable|Quantifiable|Unquantifiable|Countable|Uncountable|Numbered|Unnumbered|Calculated|Estimated|Exact|Approximate|Precise|Rough|Detailed|General|Specific|Vague|Particular|Universal|Individual|Collective|Personal|Impersonal|Private|Public|Confidential|Open|Secret|Transparent|Hidden|Concealed|Revealed|Disclosed|Undisclosed|Published|Unpublished|Announced|Unannounced|Declared|Undeclared|Stated|Unstated|Expressed|Unexpressed|Spoken|Unspoken|Verbal|Nonverbal|Oral|Written|Typed|Handwritten|Printed|Digital|Electronic|Manual|Automatic|Mechanical|Electrical|Magnetic|Optical|Acoustic|Thermal|Chemical|Physical|Virtual|Real|Actual|Theoretical|Practical|Applied|Pure|Mixed|Combined|Separated|Isolated|Integrated|Connected|Disconnected|Linked|Unlinked|Related|Unrelated|Associated|Dissociated|Attached|Detached|Bound|Unbound|Tied|Untied|Joined|Disjoined|United|Divided|Merged|Split|Fused|Separated|Blended|Distinguished|Mixed|Pure|Clean|Dirty|Fresh|Stale|New|Old|Recent|Ancient|Modern|Traditional|Contemporary|Classic|Vintage|Antique|Current|Outdated|Updated|Obsolete|Latest|Earliest|First|Last|Initial|Final|Beginning|End|Start|Finish|Opening|Closing|Top|Bottom|Upper|Lower|High|Low|Above|Below|Over|Under|Up|Down|Left|Right|Front|Back|Forward|Backward|Ahead|Behind|Before|After|Previous|Next|Prior|Following|Preceding|Succeeding|Earlier|Later|Sooner|Eventually|Immediately|Instantly|Quickly|Slowly|Rapidly|Gradually|Suddenly|Smoothly|Roughly|Gently|Harshly|Softly|Loudly|Quietly|Silently|Noisily|Peacefully|Violently|Calmly|Angrily|Happily|Sadly|Joyfully|Sorrowfully|Cheerfully|Gloomily|Hopefully|Hopelessly|Confidently|Doubtfully|Certainly|Uncertainly|Surely|Possibly|Probably|Definitely|Maybe|Perhaps|Obviously|Clearly|Apparently|Seemingly|Allegedly|Supposedly|Reportedly|Presumably|Likely|Unlikely|Probably|Improbably|Certainly|Uncertainly|Definitely|Indefinitely|Absolutely|Relatively|Completely|Partially|Totally|Partly|Fully|Halfway|Entirely|Somewhat|Mostly|Mainly|Primarily|Secondarily|Chiefly|Largely|Greatly|Significantly|Considerably|Substantially|Noticeably|Remarkably|Exceptionally|Extremely|Very|Quite|Rather|Pretty|Fairly|Reasonably|Moderately|Slightly|Barely|Hardly|Scarcely|Almost|Nearly|Approximately|Roughly|About|Around|Close|Near|Far|Distant|Away|Apart|Together|Separate|Combined|Joint|Individual|Collective|Common|Shared|Mutual|Reciprocal|Corresponding|Matching|Similar|Different|Same|Identical|Equal|Unequal|Equivalent|Nonequivalent|Comparable|Incomparable|Relative|Absolute|Conditional|Unconditional|Dependent|Independent|Related|Unrelated|Connected|Disconnected|Linked|Unlinked|Associated|Dissociated|Corresponding|Noncorresponding|Matching|Mismatched|Aligned|Misaligned|Coordinated|Uncoordinated|Synchronized|Unsynchronized|Harmonized|Disharmonized|Balanced|Unbalanced|Stable|Unstable|Steady|Unsteady|Consistent|Inconsistent|Regular|Irregular|Uniform|Nonuniform|Even|Uneven|Level|Unlevel|Flat|Curved|Straight|Bent|Direct|Indirect|Linear|Nonlinear|Circular|Angular|Round|Square|Triangular|Rectangular|Oval|Hexagonal|Octagonal|Cylindrical|Spherical|Cubic|Conical|Pyramidal|Prismatic|Geometric|Organic|Natural|Artificial|Synthetic|Genuine|Fake|Real|Imaginary|Actual|Fictional|True|False|Factual|Nonfactual|Authentic|Inauthentic|Original|Copy|Primary|Secondary|Main|Auxiliary|Principal|Subordinate|Major|Minor|Important|Unimportant|Significant|Insignificant|Essential|Nonessential|Necessary|Unnecessary|Required|Optional|Mandatory|Voluntary|Compulsory|Elective|Obligatory|Discretionary|Forced|Chosen|Imposed|Selected|Assigned|Designated|Appointed|Elected|Nominated|Recommended|Suggested|Proposed|Offered|Requested|Demanded|Required|Needed|Wanted|Desired|Preferred|Favored|Chosen|Selected|Picked|Taken|Given|Received|Accepted|Rejected|Approved|Disapproved|Endorsed|Opposed|Supported|Criticized|Praised|Blamed|Credited|Acknowledged|Ignored|Recognized|Overlooked|Noticed|Missed|Observed|Unobserved|Seen|Unseen|Viewed|Unviewed|Watched|Unwatched|Monitored|Unmonitored|Supervised|Unsupervised|Controlled|Uncontrolled|Managed|Unmanaged|Directed|Undirected|Guided|Unguided|Led|Unleaded|Conducted|Unconducted|Organized|Disorganized|Arranged|Disarranged|Structured|Unstructured|Planned|Unplanned|Designed|Undesigned|Prepared|Unprepared|Developed|Undeveloped|Created|Uncreated|Built|Unbuilt|Constructed|Deconstructed|Established|Unestablished|Founded|Unfounded|Started|Unstarted|Initiated|Uninitiated|Launched|Unlaunched|Introduced|Unintroduced|Presented|Unpresented|Delivered|Undelivered|Provided|Unprovided|Supplied|Unsupplied|Offered|Unoffered|Given|Ungiven|Granted|Ungranted|Awarded|Unawarded|Assigned|Unassigned|Allocated|Unallocated|Distributed|Undistributed|Shared|Unshared|Divided|Undivided|Split|Unsplit|Separated|Unseparated|Isolated|Unisolated|Detached|Attached|Removed|Unremoved|Taken|Untaken|Extracted|Unextracted|Withdrawn|Unwithdrawn|Pulled|Unpulled|Pushed|Unpushed|Moved|Unmoved|Transferred|Untransferred|Shifted|Unshifted|Changed|Unchanged|Modified|Unmodified|Altered|Unaltered|Adjusted|Unadjusted|Adapted|Unadapted|Transformed|Untransformed|Converted|Unconverted|Switched|Unswitched|Replaced|Unreplaced|Substituted|Unsubstituted|Exchanged|Unexchanged|Traded|Untraded|Swapped|Unswapped|Returned|Unreturned|Restored|Unrestored|Renewed|Unrenewed|Refreshed|Unrefreshed|Updated|Unupdated|Upgraded|Unupgraded|Improved|Unimproved|Enhanced|Unenhanced|Refined|Unrefined|Perfected|Unperfected|Completed|Uncompleted|Finished|Unfinished|Done|Undone|Accomplished|Unaccomplished|Achieved|Unachieved|Attained|Unattained|Reached|Unreached|Obtained|Unobtained|Acquired|Unacquired|Gained|Ungained|Earned|Unearned|Won|Lost|Succeeded|Failed|Passed|Failed|Graduated|Dropped|Qualified|Disqualified|Certified|Uncertified|Licensed|Unlicensed|Registered|Unregistered|Approved|Unapproved|Accepted|Unaccepted|Admitted|Unadmitted|Enrolled|Unenrolled|Subscribed|Unsubscribed|Joined|Unjoined|Participated|Nonparticipated|Attended|Unattended|Visited|Unvisited|Toured|Untoured|Traveled|Untraveled|Moved|Unmoved|Relocated|Unrelocated|Transferred|Untransferred|Shifted|Unshifted|Changed|Unchanged|Switched|Unswitched|Converted|Unconverted|Transformed|Untransformed|Adapted|Unadapted|Adjusted|Unadjusted|Modified|Unmodified|Altered|Unaltered|Updated|Unupdated|Upgraded|Unupgraded|Improved|Unimproved|Enhanced|Unenhanced|Refined|Unrefined|Polished|Unpolished|Perfected|Unperfected|Completed|Uncompleted|Finalized|Unfinalized|Concluded|Unconcluded|Ended|Unended|Terminated|Unterminated|Stopped|Unstopped|Ceased|Unceased|Discontinued|Continued|Suspended|Resumed|Paused|Unpaused|Interrupted|Uninterrupted|Delayed|Undelayed|Postponed|Unpostponed|Rescheduled|Unrescheduled|Extended|Unextended|Prolonged|Unprolonged|Shortened|Unshortened|Reduced|Unreduced|Decreased|Undecreased|Minimized|Unminimized|Limited|Unlimited|Restricted|Unrestricted|Constrained|Unconstrained|Bounded|Unbounded|Contained|Uncontained|Enclosed|Unenclosed|Surrounded|Unsurrounded|Protected|Unprotected|Secured|Unsecured|Guarded|Unguarded|Defended|Undefended|Shielded|Unshielded|Covered|Uncovered|Hidden|Unhidden|Concealed|Unconcealed|Masked|Unmasked|Disguised|Undisguised|Camouflaged|Uncamouflaged|Obscured|Unobscured|Blocked|Unblocked|Prevented|Unprevented|Avoided|Unavoided|Evaded|Unevaded|Escaped|Unescaped|Fled|Unfled|Retreated|Unretrieved|Withdrawn|Unwithdwarn|Removed|Unremoved|Eliminated|Uneliminated|Deleted|Undeleted|Erased|Unerased|Cleared|Uncleared|Cleaned|Uncleaned|Purged|Unpurged|Flushed|Unflushed|Emptied|Unemptied|Vacated|Unvacated|Abandoned|Unabandoned|Deserted|Undeserted|Left|Unleft|Departed|Undeparted|Exited|Unexited|Quit|Unquit|Resigned|Unresigned|Retired|Unretired|Stepped|Unstepped|Backed|Unbacked|Returned|Unreturned|Came|Uncame|Arrived|Unarrived|Entered|Unentered|Joined|Unjoined|Participated|Unparticipated|Engaged|Unengaged|Involved|Uninvolved|Included|Excluded|Incorporated|Unincorporated|Integrated|Unintegrated|Combined|Uncombined|Merged|Unmerged|United|Disunited|Connected|Disconnected|Linked|Unlinked|Attached|Unattached|Bound|Unbound|Tied|Untied|Secured|Unsecured|Fastened|Unfastened|Fixed|Unfixed|Stable|Unstable|Steady|Unsteady|Consistent|Inconsistent|Reliable|Unreliable|Dependable|Undependable|Trustworthy|Untrustworthy|Credible|Incredible|Believable|Unbelievable|Convincing|Unconvincing|Persuasive|Unpersuasive|Compelling|Uncompelling|Attractive|Unattractive|Appealing|Unappealing|Interesting|Uninteresting|Engaging|Unengaging|Entertaining|Unentertainting|Enjoyable|Unenjoyable|Pleasant|Unpleasant|Comfortable|Uncomfortable|Convenient|Inconvenient|Easy|Difficult|Simple|Complex|Straightforward|Complicated|Clear|Unclear|Obvious|Unobvious|Evident|Unevident|Apparent|Unapparent|Visible|Invisible|Noticeable|Unnoticeable|Remarkable|Unremarkable|Notable|Unnotable|Significant|Insignificant|Important|Unimportant|Major|Minor|Small|Large|Big|Little|Huge|Tiny|Enormous|Minute|Massive|Microscopic|Gigantic|Miniscule|Colossal|Infinitesimal|Tremendous|Negligible|Immense|Small|Vast|Limited|Extensive|Restricted|Comprehensive|Incomplete|Complete|Partial|Full|Empty|Total|Fractional|Whole|Broken|Entire|Fragmented|All|None|Every|No|Each|Neither|Some|Any|Few|Several|Many|Much|Most|Least|More|Less|Greater|Smaller|Higher|Lower|Bigger|Smaller|Larger|Tinier|Wider|Narrower|Longer|Shorter|Taller|Shorter|Deeper|Shallower|Thicker|Thinner|Heavier|Lighter|Stronger|Weaker|Harder|Softer|Tougher|Gentler|Rougher|Smoother|Coarser|Finer|Denser|Sparser|Tighter|Looser|Firmer|Softer|Stiffer|More Flexible|More Rigid|More Elastic|Less Elastic|More Durable|Less Durable|More Fragile|Less Fragile|More Stable|Less Stable|More Secure|Less Secure|Safer|More Dangerous|More Risky|Less Risky|More Hazardous|Less Hazardous|More Threatening|Less Threatening|More Alarming|Less Alarming|More Calming|Less Calming|More Exciting|Less Exciting|More Boring|Less Boring|More Thrilling|Less Thrilling|More Interesting|Less Interesting|More Engaging|Less Engaging|More Captivating|Less Captivating|More Attractive|Less Attractive|More Beautiful|Less Beautiful|More Handsome|Less Handsome|More Gorgeous|Less Gorgeous|More Stunning|Less Stunning|More Impressive|Less Impressive|More Spectacular|Less Spectacular|More Magnificent|Less Magnificent|More Splendid|Less Splendid|More Superb|Less Superb|More Excellent|Less Excellent|More Outstanding|Less Outstanding|More Exceptional|Less Exceptional|More Extraordinary|Less Extraordinary|More Special|Less Special|More Unique|Less Unique|More Rare|Less Rare|More Common|Less Common|More Frequent|Less Frequent|More Often|Less Often|More Regular|Less Regular|More Constant|Less Constant|More Continuous|Less Continuous|More Permanent|Less Permanent|More Temporary|Less Temporary|More Brief|Less Brief|More Extended|Less Extended|More Prolonged|Less Prolonged|More Quick|Less Quick|More Slow|Less Slow|More Fast|Less Fast|More Rapid|Less Rapid|More Gradual|Less Gradual|More Sudden|Less Sudden|More Immediate|Less Immediate|More Delayed|Less Delayed|More Prompt|Less Prompt|More Late|Less Late|More Early|Less Early|More Timely|Less Timely|More Scheduled|Less Scheduled|More Planned|Less Planned|More Expected|Less Expected|More Predictable|Less Predictable|More Anticipated|Less Anticipated|More Prepared|Less Prepared|More Ready|Less Ready|More Available|Less Available|More Present|Less Present|More Nearby|Less Nearby|More Close|Less Close|More Distant|Less Distant|More Local|Less Local|More Foreign|Less Foreign|More Domestic|Less Domestic|More International|Less International|More National|Less National|More Regional|Less Regional|More Global|Less Global|More Universal|Less Universal|More Limited|Less Limited|More Restricted|Less Restricted|More Bounded|Less Bounded|More Finite|Less Finite|More Measurable|Less Measurable|More Exact|Less Exact|More Precise|Less Precise|More Accurate|Less Accurate|More Detailed|Less Detailed|More Specific|Less Specific|More General|Less General|More Particular|Less Particular|More Individual|Less Individual|More Personal|Less Personal|More Private|Less Private|More Public|Less Public|More Confidential|Less Confidential|More Secret|Less Secret|More Hidden|Less Hidden|More Concealed|Less Concealed|More Revealed|Less Revealed|More Disclosed|Less Disclosed|More Published|Less Published|More Announced|Less Announced|More Declared|Less Declared|More Stated|Less Stated|More Expressed|Less Expressed|More Spoken|Less Spoken|More Written|Less Written|More Digital|Less Digital|More Electronic|Less Electronic|More Manual|Less Manual|More Automatic|Less Automatic|More Mechanical|Less Mechanical|More Physical|Less Physical|More Virtual|Less Virtual|More Real|Less Real|More Actual|Less Actual|More Theoretical|Less Theoretical|More Practical|Less Practical|More Applied|Less Applied|More Pure|Less Pure|More Mixed|Less Mixed|More Combined|Less Combined|More Separated|Less Separated|More Isolated|Less Isolated|More Integrated|Less Integrated|More Connected|Less Connected|More Linked|Less Linked|More Related|Less Related|More Associated|Less Associated|More Attached|Less Attached|More Bound|Less Bound|More Joined|Less Joined|More United|Less United|More Merged|Less Merged|More Blended|Less Blended|More Distinguished|Less Distinguished|More Clean|Less Clean|More Dirty|Less Dirty|More Fresh|Less Fresh|More Stale|Less Stale|More New|Less New|More Old|Less Old|More Recent|Less Recent|More Ancient|Less Ancient|More Modern|Less Modern|More Traditional|Less Traditional|More Contemporary|Less Contemporary|More Classic|Less Classic|More Current|Less Current|More Updated|Less Updated|More Latest|Less Latest|More First|Less First|More Last|Less Last|More Initial|Less Initial|More Final|Less Final|More Beginning|Less Beginning|More End|Less End|More Top|Less Top|More Bottom|Less Bottom|More Upper|Less Upper|More Lower|Less Lower|More High|Less High|More Low|Less Low|More Above|Less Above|More Below|Less Below|More Over|Less Over|More Under|Less Under|More Front|Less Front|More Back|Less Back|More Forward|Less Forward|More Backward|Less Backward|More Ahead|Less Ahead|More Behind|Less Behind|More Before|Less Before|More After|Less After|More Previous|Less Previous|More Next|Less Next|More Prior|Less Prior|More Following|Less Following|More Preceding|Less Preceding|More Earlier|Less Earlier|More Later|Less Later|More Sooner|Less Sooner|More Immediately|Less Immediately|More Quickly|Less Quickly|More Slowly|Less Slowly|More Rapidly|Less Rapidly|More Gradually|Less Gradually|More Suddenly|Less Suddenly|More Smoothly|Less Smoothly|More Roughly|Less Roughly|More Gently|Less Gently|More Harshly|Less Harshly|More Softly|Less Softly|More Loudly|Less Loudly|More Quietly|Less Quietly|More Silently|Less Silently|More Noisily|Less Noisily|More Peacefully|Less Peacefully|More Violently|Less Violently|More Calmly|Less Calmly|More Angrily|Less Angrily|More Happily|Less Happily|More Sadly|Less Sadly|More Joyfully|Less Joyfully|More Cheerfully|Less Cheerfully|More Hopefully|Less Hopefully|More Confidently|Less Confidently|More Certainly|Less Certainly|More Probably|Less Probably|More Definitely|Less Definitely|More Obviously|Less Obviously|More Clearly|Less Clearly|More Apparently|Less Apparently|More Likely|Less Likely|More Absolutely|Less Absolutely|More Completely|Less Completely|More Partially|Less Partially|More Totally|Less Totally|More Fully|Less Fully|More Entirely|Less Entirely|More Mostly|Less Mostly|More Mainly|Less Mainly|More Primarily|Less Primarily|More Chiefly|Less Chiefly|More Largely|Less Largely|More Greatly|Less Greatly|More Significantly|Less Significantly|More Considerably|Less Considerably|More Substantially|Less Substantially|More Noticeably|Less Noticeably|More Remarkably|Less Remarkably|More Exceptionally|Less Exceptionally|More Extremely|Less Extremely|More Very|Less Very|More Quite|Less Quite|More Rather|Less Rather|More Pretty|Less Pretty|More Fairly|Less Fairly|More Reasonably|Less Reasonably|More Moderately|Less Moderately|More Slightly|Less Slightly|More Barely|Less Barely|More Hardly|Less Hardly|More Almost|Less Almost|More Nearly|Less Nearly|More Approximately|Less Approximately|More Roughly|Less Roughly|More About|Less About|More Around|Less Around|More Close|Less Close|More Together|Less Together|More Apart|Less Apart|More Similar|Less Similar|More Different|Less Different|More Same|Less Same|More Equal|Less Equal|More Equivalent|Less Equivalent|More Comparable|Less Comparable|More Relative|Less Relative|More Conditional|Less Conditional|More Dependent|Less Dependent|More Independent|Less Independent|More Related|Less Related|More Connected|Less Connected|More Associated|Less Associated|More Corresponding|Less Corresponding|More Matching|Less Matching|More Aligned|Less Aligned|More Coordinated|Less Coordinated|More Synchronized|Less Synchronized|More Balanced|Less Balanced|More Stable|Less Stable|More Consistent|Less Consistent|More Regular|Less Regular|More Uniform|Less Uniform|More Even|Less Even|More Level|Less Level|More Flat|Less Flat|More Straight|Less Straight|More Direct|Less Direct|More Linear|Less Linear|More Circular|Less Circular|More Round|Less Round|More Square|Less Square|More Natural|Less Natural|More Artificial|Less Artificial|More Genuine|Less Genuine|More Real|Less Real|More True|Less True|More Authentic|Less Authentic|More Original|Less Original|More Primary|Less Primary|More Main|Less Main|More Principal|Less Principal|More Major|Less Major|More Important|Less Important|More Significant|Less Significant|More Essential|Less Essential|More Necessary|Less Necessary|More Required|Less Required|More Mandatory|Less Mandatory|More Compulsory|Less Compulsory|More Obligatory|Less Obligatory|More Forced|Less Forced|More Chosen|Less Chosen|More Selected|Less Selected|More Preferred|Less Preferred|More Favored|Less Favored|More Accepted|Less Accepted|More Approved|Less Approved|More Endorsed|Less Endorsed|More Supported|Less Supported|More Praised|Less Praised|More Credited|Less Credited|More Acknowledged|Less Acknowledged|More Recognized|Less Recognized|More Noticed|Less Noticed|More Observed|Less Observed|More Seen|Less Seen|More Viewed|Less Viewed|More Watched|Less Watched|More Monitored|Less Monitored|More Supervised|Less Supervised|More Controlled|Less Controlled|More Managed|Less Managed|More Directed|Less Directed|More Guided|Less Guided|More Led|Less Led|More Conducted|Less Conducted|More Organized|Less Organized|More Arranged|Less Arranged|More Structured|Less Structured|More Planned|Less Planned|More Designed|Less Designed|More Prepared|Less Prepared|More Developed|Less Developed|More Created|Less Created|More Built|Less Built|More Constructed|Less Constructed|More Established|Less Established|More Founded|Less Founded|More Started|Less Started|More Initiated|Less Initiated|More Launched|Less Launched|More Introduced|Less Introduced|More Presented|Less Presented|More Delivered|Less Delivered|More Provided|Less Provided|More Supplied|Less Supplied|More Offered|Less Offered|More Given|Less Given|More Granted|Less Granted|More Awarded|Less Awarded|More Assigned|Less Assigned|More Allocated|Less Allocated|More Distributed|Less Distributed|More Shared|Less Shared|More Divided|Less Divided|More Separated|Less Separated|More Isolated|Less Isolated|More Removed|Less Removed|More Taken|Less Taken|More Extracted|Less Extracted|More Withdrawn|Less Withdrawn|More Moved|Less Moved|More Transferred|Less Transferred|More Shifted|Less Shifted|More Changed|Less Changed|More Modified|Less Modified|More Altered|Less Altered|More Adjusted|Less Adjusted|More Adapted|Less Adapted|More Transformed|Less Transformed|More Converted|Less Converted|More Switched|Less Switched|More Replaced|Less Replaced|More Returned|Less Returned|More Restored|Less Restored|More Renewed|Less Renewed|More Updated|Less Updated|More Upgraded|Less Upgraded|More Improved|Less Improved|More Enhanced|Less Enhanced|More Refined|Less Refined|More Perfected|Less Perfected|More Completed|Less Completed|More Finished|Less Finished|More Accomplished|Less Accomplished|More Achieved|Less Achieved|More Attained|Less Attained|More Obtained|Less Obtained|More Acquired|Less Acquired|More Gained|Less Gained|More Earned|Less Earned|More Won|Less Won|More Succeeded|Less Succeeded|More Passed|Less Passed|More Qualified|Less Qualified|More Certified|Less Certified|More Licensed|Less Licensed|More Registered|Less Registered|More Approved|Less Approved|More Accepted|Less Accepted|More Admitted|Less Admitted|More Enrolled|Less Enrolled|More Joined|Less Joined|More Participated|Less Participated|More Attended|Less Attended|More Visited|Less Visited|More Traveled|Less Traveled|More Moved|Less Moved|More Relocated|Less Relocated|More Transferred|Less Transferred|More Changed|Less Changed|More Switched|Less Switched|More Converted|Less Converted|More Transformed|Less Transformed|More Adapted|Less Adapted|More Adjusted|Less Adjusted|More Modified|Less Modified|More Updated|Less Updated|More Improved|Less Improved|More Enhanced|Less Enhanced|More Completed|Less Completed|More Finished|Less Finished|More Ended|Less Ended|More Stopped|Less Stopped|More Ceased|Less Ceased|More Suspended|Less Suspended|More Paused|Less Paused|More Delayed|Less Delayed|More Extended|Less Extended|More Reduced|Less Reduced|More Limited|Less Limited|More Restricted|Less Restricted|More Protected|Less Protected|More Secured|Less Secured|More Hidden|Less Hidden|More Revealed|Less Revealed|More Disclosed|Less Disclosed|More Published|Less Published|More Announced|Less Announced|More Declared|Less Declared|More Stated|Less Stated|More Expressed|Less Expressed|More Communicated|Less Communicated|More Transmitted|Less Transmitted|More Sent|Less Sent|More Received|Less Received|More Delivered|Less Delivered|More Forwarded|Less Forwarded|More Copied|Less Copied|More Saved|Less Saved|More Stored|Less Stored|More Archived|Less Archived|More Backed|Less Backed|More Recovered|Less Recovered|More Restored|Less Restored|More Retrieved|Less Retrieved|More Found|Less Found|More Located|Less Located|More Discovered|Less Discovered|More Identified|Less Identified|More Recognized|Less Recognized|More Detected|Less Detected|More Searched|Less Searched|More Explored|Less Explored|More Investigated|Less Investigated|More Examined|Less Examined|More Inspected|Less Inspected|More Analyzed|Less Analyzed|More Evaluated|Less Evaluated|More Assessed|Less Assessed|More Measured|Less Measured|More Calculated|Less Calculated|More Computed|Less Computed|More Estimated|Less Estimated|More Predicted|Less Predicted|More Forecasted|Less Forecasted|More Projected|Less Projected|More Planned|Less Planned|More Scheduled|Less Scheduled|More Arranged|Less Arranged|More Organized|Less Organized|More Coordinated|Less Coordinated|More Managed|Less Managed|More Supervised|Less Supervised|More Controlled|Less Controlled|More Directed|Less Directed|More Guided|Less Guided|More Led|Less Led|More Administered|Less Administered|More Operated|Less Operated|More Executed|Less Executed|More Performed|Less Performed|More Conducted|Less Conducted|More Implemented|Less Implemented|More Applied|Less Applied|More Used|Less Used|More Utilized|Less Utilized|More Employed|Less Employed|More Deployed|Less Deployed|More Installed|Less Installed|More Configured|Less Configured|More Set|Less Set|More Established|Less Established|More Created|Less Created|More Built|Less Built|More Developed|Less Developed|More Designed|Less Designed|More Constructed|Less Constructed|More Assembled|Less Assembled|More Manufactured|Less Manufactured|More Produced|Less Produced|More Generated|Less Generated|More Made|Less Made|More Formed|Less Formed|More Shaped|Less Shaped|More Molded|Less Molded|More Crafted|Less Crafted|More Fashioned|Less Fashioned|More Fabricated|Less Fabricated|More Constructed|Less Constructed|More Erected|Less Erected|More Raised|Less Raised|More Lifted|Less Lifted|More Elevated|Less Elevated|More Hoisted|Less Hoisted|More Suspended|Less Suspended|More Hung|Less Hung|More Mounted|Less Mounted|More Attached|Less Attached|More Fixed|Less Fixed|More Secured|Less Secured|More Fastened|Less Fastened|More Tied|Less Tied|More Bound|Less Bound|More Connected|Less Connected|More Linked|Less Linked|More Joined|Less Joined|More United|Less United|More Combined|Less Combined|More Merged|Less Merged|More Integrated|Less Integrated|More Incorporated|Less Incorporated|More Included|Less Included|More Involved|Less Involved|More Engaged|Less Engaged|More Participated|Less Participated|More Contributed|Less Contributed|More Assisted|Less Assisted|More Helped|Less Helped|More Supported|Less Supported|More Aided|Less Aided|More Facilitated|Less Facilitated|More Enabled|Less Enabled|More Empowered|Less Empowered|More Strengthened|Less Strengthened|More Reinforced|Less Reinforced|More Enhanced|Less Enhanced|More Improved|Less Improved|More Upgraded|Less Upgraded|More Advanced|Less Advanced|More Developed|Less Developed|More Evolved|Less Evolved|More Progressed|Less Progressed|More Grown|Less Grown|More Expanded|Less Expanded|More Extended|Less Extended|More Enlarged|Less Enlarged|More Increased|Less Increased|More Multiplied|Less Multiplied|More Amplified|Less Amplified|More Boosted|Less Boosted|More Raised|Less Raised|More Elevated|Less Elevated|More Lifted|Less Lifted|More Heightened|Less Heightened|More Intensified|Less Intensified|More Strengthened|Less Strengthened|More Powered|Less Powered|More Energized|Less Energized|More Activated|Less Activated|More Stimulated|Less Stimulated|More Motivated|Less Motivated|More Inspired|Less Inspired|More Encouraged|Less Encouraged|More Supported|Less Supported|More Backed|Less Backed|More Endorsed|Less Endorsed|More Approved|Less Approved|More Accepted|Less Accepted|More Welcomed|Less Welcomed|More Embraced|Less Embraced|More Adopted|Less Adopted|More Taken|Less Taken|More Chosen|Less Chosen|More Selected|Less Selected|More Picked|Less Picked|More Preferred|Less Preferred|More Favored|Less Favored|More Liked|Less Liked|More Loved|Less Loved|More Adored|Less Adored|More Cherished|Less Cherished|More Valued|Less Valued|More Appreciated|Less Appreciated|More Respected|Less Respected|More Admired|Less Admired|More Honored|Less Honored|More Revered|Less Revered|More Esteemed|Less Esteemed|More Regarded|Less Regarded|More Considered|Less Considered|More Thought|Less Thought|More Believed|Less Believed|More Trusted|Less Trusted|More Relied|Less Relied|More Depended|Less Depended|More Counted|Less Counted|More Expected|Less Expected|More Anticipated|Less Anticipated|More Predicted|Less Predicted|More Forecasted|Less Forecasted|More Projected|Less Projected|More Estimated|Less Estimated|More Calculated|Less Calculated|More Figured|Less Figured|More Determined|Less Determined|More Decided|Less Decided|More Resolved|Less Resolved|More Concluded|Less Concluded|More Settled|Less Settled|More Fixed|Less Fixed|More Established|Less Established|More Confirmed|Less Confirmed|More Verified|Less Verified|More Validated|Less Validated|More Certified|Less Certified|More Authenticated|Less Authenticated|More Authorized|Less Authorized|More Approved|Less Approved|More Permitted|Less Permitted|More Allowed|Less Allowed|More Granted|Less Granted|More Given|Less Given|More Provided|Less Provided|More Supplied|Less Supplied|More Furnished|Less Furnished|More Delivered|Less Delivered|More Offered|Less Offered|More Presented|Less Presented|More Shown|Less Shown|More Displayed|Less Displayed|More Exhibited|Less Exhibited|More Demonstrated|Less Demonstrated|More Revealed|Less Revealed|More Exposed|Less Exposed|More Uncovered|Less Uncovered|More Discovered|Less Discovered|More Found|Less Found|More Located|Less Located|More Identified|Less Identified|More Recognized|Less Recognized|More Detected|Less Detected|More Noticed|Less Noticed|More Observed|Less Observed|More Seen|Less Seen|More Viewed|Less Viewed|More Watched|Less Watched|More Looked|Less Looked|More Examined|Less Examined|More Inspected|Less Inspected|More Investigated|Less Investigated|More Explored|Less Explored|More Searched|Less Searched|More Studied|Less Studied|More Researched|Less Researched|More Analyzed|Less Analyzed|More Evaluated|Less Evaluated|More Assessed|Less Assessed|More Reviewed|Less Reviewed|More Checked|Less Checked|More Tested|Less Tested|More Tried|Less Tried|More Attempted|Less Attempted|More Undertaken|Less Undertaken|More Performed|Less Performed|More Executed|Less Executed|More Carried|Less Carried|More Conducted|Less Conducted|More Implemented|Less Implemented|More Applied|Less Applied|More Practiced|Less Practiced|More Exercised|Less Exercised|More Used|Less Used|More Utilized|Less Utilized|More Employed|Less Employed|More Operated|Less Operated|More Handled|Less Handled|More Managed|Less Managed|More Controlled|Less Controlled|More Directed|Less Directed|More Supervised|Less Supervised|More Overseen|Less Overseen|More Monitored|Less Monitored|More Tracked|Less Tracked|More Followed|Less Followed|More Pursued|Less Pursued|More Chased|Less Chased|More Hunted|Less Hunted|More Sought|Less Sought|More Looked|Less Looked|More Searched|Less Searched|More Found|Less Found|More Located|Less Located|More Discovered|Less Discovered|More Uncovered|Less Uncovered|More Revealed|Less Revealed|More Exposed|Less Exposed|More Shown|Less Shown|More Displayed|Less Displayed|More Presented|Less Presented|More Demonstrated|Less Demonstrated|More Illustrated|Less Illustrated|More Explained|Less Explained|More Described|Less Described|More Detailed|Less Detailed|More Specified|Less Specified|More Clarified|Less Clarified|More Defined|Less Defined|More Outlined|Less Outlined|More Summarized|Less Summarized|More Listed|Less Listed|More Enumerated|Less Enumerated|More Catalogued|Less Catalogued|More Indexed|Less Indexed|More Recorded|Less Recorded|More Documented|Less Documented|More Noted|Less Noted|More Written|Less Written|More Typed|Less Typed|More Printed|Less Printed|More Published|Less Published|More Released|Less Released|More Issued|Less Issued|More Distributed|Less Distributed|More Circulated|Less Circulated|More Spread|Less Spread|More Shared|Less Shared|More Communicated|Less Communicated|More Transmitted|Less Transmitted|More Sent|Less Sent|More Delivered|Less Delivered|More Forwarded|Less Forwarded|More Passed|Less Passed|More Handed|Less Handed|More Given|Less Given|More Provided|Less Provided|More Supplied|Less Supplied|More Offered|Less Offered|More Presented|Less Presented|More Granted|Less Granted|More Awarded|Less Awarded|More Assigned|Less Assigned|More Allocated|Less Allocated|More Allotted|Less Allotted|More Designated|Less Designated|More Appointed|Less Appointed|More Named|Less Named|More Chosen|Less Chosen|More Selected|Less Selected|More Picked|Less Picked|More Elected|Less Elected|More Voted|Less Voted|More Nominated|Less Nominated|More Recommended|Less Recommended|More Suggested|Less Suggested|More Proposed|Less Proposed|More Offered|Less Offered|More Requested|Less Requested|More Asked|Less Asked|More Demanded|Less Demanded|More Required|Less Required|More Needed|Less Needed|More Wanted|Less Wanted|More Desired|Less Desired|More Sought|Less Sought|More Wished|Less Wished|More Hoped|Less Hoped|More Expected|Less Expected|More Anticipated|Less Anticipated|More Awaited|Less Awaited|More Looked|Less Looked|More Waited|Less Waited)\b', re.IGNORECASE),
            ],
            'companies': [
                re.compile(r'\b(?:Date|Time|Page|Total|Amount|Number|Code|ID|Reference|Address|Phone|Email|Website|Report|Document|File|Version|Copy|Draft|Final|January|February|March|April|May|June|July|August|September|October|November|December|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Morning|Afternoon|Evening|Night|Today|Tomorrow|Yesterday|Week|Month|Year|Quarter|Annual|Daily|Weekly|Monthly|Quarterly|Hourly|Minutes|Seconds|Hours|Days|Weeks|Months|Years|AM|PM|UTC|GMT|EST|PST|CST|MST|PHT|JST|KST|SGT|MYT|THT|IDT|VNT|HKT|AUT|CAT|CHT|SET|NOT|DKT|PLT|CZT|HUT|ZAT|BRT|MXT|ART|CLT|COT|PET|UYT|BOT|PYT|INT|PKT|BDT|LKT|NPT|BTT|MVT|AFT|MMT|KHT|LAT|BNT)\b'),
            ],
            'locations': [
                re.compile(r'\b(?:Date|Time|Page|Total|Amount|Number|Code|ID|Reference|Subject|Phone|Email|Website|Report|Document|File|Version|Copy|Draft|Final|Mr|Ms|Mrs|Dr|Prof|Sir|Madam|Miss|Atty|Engr|Archt)\b', re.IGNORECASE),
            ]
        }
        
        # Validation patterns for entities
        self.validation_patterns = {
            'people': [
                re.compile(r'^[A-Z][a-z]+ [A-Z][a-z]+$'),  # Basic First Last format
                re.compile(r'^[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+$'),  # First M. Last format
                re.compile(r'^[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+$'),  # First Middle Last
            ],
            'companies': [
                re.compile(r'[A-Za-z]'),  # Must contain at least one letter
            ],
            'locations': [
                re.compile(r'[A-Za-z]'),  # Must contain at least one letter
            ]
        }
    
    def _initialize_models(self):
        """Initialize models once for reuse"""
        try:
            with self._model_lock:
                if self._ocr_processor is None:
                    self._ocr_processor = OCRProcessor(
                        languages=['en'],
                        gpu=not os.environ.get('FORCE_CPU_ONLY', False),
                        verbose=False
                    )
                
                if self._data_extractor is None:
                    device = "cuda" if torch.cuda.is_available() and not os.environ.get('FORCE_CPU_ONLY') else "cpu"
                    self._data_extractor = QwenExtractor(device=device)
            
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise
    
    def _is_false_positive(self, entity, entity_type):
        """Check if entity is a false positive"""
        if entity_type in self.false_positive_patterns:
            for pattern in self.false_positive_patterns[entity_type]:
                if pattern.search(entity):
                    return True
        return False
    
    def _is_valid_entity(self, entity, entity_type):
        """Validate entity based on type-specific rules"""
        if entity_type == 'people':
            # Check length (names shouldn't be too long or too short)
            if len(entity) < 4 or len(entity) > 50:
                return False
            
            # Must have at least 2 words
            words = entity.split()
            if len(words) < 2:
                return False
            
            # Check for common business terms that might be misclassified
            business_terms = ['Inc', 'Corp', 'LLC', 'Ltd', 'Co', 'Company', 'Corporation', 'Technologies', 'Solutions', 'Systems', 'Services']
            if any(term in entity for term in business_terms):
                return False
            
            # Check if it looks like a person name using validation patterns
            for pattern in self.validation_patterns['people']:
                if pattern.match(entity):
                    return True
            
            # Additional validation for Filipino names
            if any(word in entity.lower() for word in ['de', 'del', 'delos', 'dela', 'de la']):
                return True
            
            return False
            
        elif entity_type == 'companies':
            # Check length
            if len(entity) < 2 or len(entity) > 100:
                return False
            
            # Must contain at least one letter
            if not any(c.isalpha() for c in entity):
                return False
            
            # Check if it's just numbers or very short acronym without context
            if len(entity) <= 2 and entity.isupper():
                return False
            
            return True
            
        elif entity_type == 'locations':
            # Check length
            if len(entity) < 2 or len(entity) > 80:
                return False
            
            # Must contain at least one letter
            if not any(c.isalpha() for c in entity):
                return False
            
            return True
            
        return True
    
    def _extract_entities(self, text):
        """Extract people, companies, and locations from text with improved accuracy"""
        entities = {
            "people": [],
            "companies": [],
            "locations": []
        }
        
        # Extract people
        for pattern in self.person_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1] if len(match) > 1 else ""
                
                if match and not self._is_false_positive(match, 'people') and self._is_valid_entity(match, 'people'):
                    entities["people"].append(match.strip())
        
        # Extract companies
        for pattern in self.company_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1] if len(match) > 1 else ""
                
                if match and not self._is_false_positive(match, 'companies') and self._is_valid_entity(match, 'companies'):
                    entities["companies"].append(match.strip())
        
        # Extract locations
        for pattern in self.location_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1] if len(match) > 1 else ""
                
                if match and not self._is_false_positive(match, 'locations') and self._is_valid_entity(match, 'locations'):
                    entities["locations"].append(match.strip())
        
        # Remove duplicates while preserving order and clean up
        for key in entities:
            # Remove duplicates
            seen = set()
            unique_entities = []
            for item in entities[key]:
                item_lower = item.lower()
                if item_lower not in seen:
                    seen.add(item_lower)
                    unique_entities.append(item)
            
            # Final filtering
            entities[key] = [item for item in unique_entities if item and len(item.strip()) > 1]
        
        return entities
    
    def _extract_dates_and_amounts(self, text):
        """Extract dates and amounts from text with improved patterns"""
        dates = set()
        amounts = set()
        
        # Extract dates
        for pattern in self.date_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1] if len(match) > 1 else ""
                
                if match and len(match.strip()) > 3:
                    # Clean up the date
                    cleaned_date = re.sub(r'\s+', ' ', match.strip())
                    dates.add(cleaned_date)
        
        # Extract amounts
        for pattern in self.amount_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1] if len(match) > 1 else ""
                
                if match and len(match.strip()) > 0:
                    # Clean up the amount
                    cleaned_amount = re.sub(r'\s+', ' ', match.strip())
                    amounts.add(cleaned_amount)
        
        return list(dates), list(amounts)
    
    def _determine_document_type(self, text, filename):
        """Dynamically determine document type from content and filename"""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        type_indicators = {
            'invoice': ['invoice', 'bill', 'billing', 'amount due', 'payment due', 'invoice no', 'bill no'],
            'receipt': ['receipt', 'or number', 'official receipt', 'acknowledgment receipt', 'cash receipt'],
            'contract': ['agreement', 'contract', 'terms and conditions', 'whereas', 'party of the first part', 'party of the second part'],
            'letter': ['dear', 'sincerely', 'regards', 'yours truly', 'yours faithfully', 'best regards'],
            'email': ['from:', 'to:', 'subject:', 'sent:', '@', 'cc:', 'bcc:', 're:'],
            'financial_statement': ['balance sheet', 'income statement', 'profit and loss', 'cash flow statement', 'statement of financial position'],
            'purchase_order': ['purchase order', 'po number', 'purchase requisition', 'pr number'],
            'voucher': ['voucher', 'check number', 'cheque number', 'disbursement voucher', 'payment voucher'],
            'report': ['report', 'analysis', 'summary', 'findings', 'conclusion', 'executive summary'],
            'memo': ['memorandum', 'memo', 'circular', 'office memorandum', 'internal memo'],
            'certificate': ['certificate', 'certification', 'certified', 'certificate of', 'this is to certify'],
            'delivery_receipt': ['delivery receipt', 'dr number', 'delivered to', 'received by'],
            'quotation': ['quotation', 'quote', 'price quote', 'quotation no', 'rfq'],
            'payroll': ['payroll', 'salary', 'wages', 'payslip', 'pay stub', 'compensation'],
            'tax_document': ['tax', 'bir', 'form 2307', 'withholding', 'vat', 'ewt'],
            'bank_statement': ['bank statement', 'account statement', 'balance inquiry', 'transaction history'],
            'insurance': ['insurance', 'policy', 'premium', 'coverage', 'claim', 'insured'],
            'lease_agreement': ['lease', 'rent', 'tenant', 'landlord', 'monthly rental'],
            'employment': ['employment', 'job', 'position', 'employee', 'employer', 'salary agreement'],
        }
        
        # Check filename first (higher weight)
        filename_scores = {}
        for doc_type, keywords in type_indicators.items():
            score = sum(2 for keyword in keywords if keyword in filename_lower)
            if score > 0:
                filename_scores[doc_type] = score
        
        # Check content
        content_scores = {}
        for doc_type, keywords in type_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                content_scores[doc_type] = score
        
        # Combine scores (filename gets double weight)
        all_scores = {}
        for doc_type in set(list(filename_scores.keys()) + list(content_scores.keys())):
            all_scores[doc_type] = filename_scores.get(doc_type, 0) + content_scores.get(doc_type, 0)
        
        if all_scores:
            best_type = max(all_scores, key=all_scores.get)
            # Only return if score is reasonable
            if all_scores[best_type] >= 1:
                return best_type
        
        return 'other'
    
    def _extract_key_information(self, text, document_type, entities):
        """Extract key information based on document type"""
        key_info = {}
        
        if document_type == 'invoice':
            key_info.update(self._extract_invoice_info(text))
        elif document_type == 'contract':
            key_info.update(self._extract_contract_info(text))
        elif document_type == 'email':
            key_info.update(self._extract_email_info(text))
        elif document_type == 'financial_statement':
            key_info.update(self._extract_financial_info(text))
        elif document_type == 'report':
            key_info.update(self._extract_report_info(text))
        elif document_type == 'receipt':
            key_info.update(self._extract_receipt_info(text))
        elif document_type == 'purchase_order':
            key_info.update(self._extract_purchase_order_info(text))
        elif document_type == 'delivery_receipt':
            key_info.update(self._extract_delivery_receipt_info(text))
        elif document_type == 'payroll':
            key_info.update(self._extract_payroll_info(text))
        else:
            # Generic extraction for unknown types
            key_info.update(self._extract_generic_info(text, entities))
        
        return key_info
    
    def _extract_invoice_info(self, text):
        """Extract invoice-specific information"""
        info = {}
        
        # Invoice number patterns
        invoice_patterns = [
            re.compile(r'(?:invoice|bill)\s*#?\s*:?\s*([A-Z0-9\-]+)', re.IGNORECASE),
            re.compile(r'invoice\s+(?:no|number)\s*:?\s*([A-Z0-9\-]+)', re.IGNORECASE),
            re.compile(r'inv\s*#?\s*:?\s*([A-Z0-9\-]+)', re.IGNORECASE),
        ]
        
        for pattern in invoice_patterns:
            match = pattern.search(text)
            if match:
                info['invoice_number'] = match.group(1).strip()
                break
        
        # Due date patterns
        due_patterns = [
            re.compile(r'due\s+date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})', re.IGNORECASE),
            re.compile(r'payment\s+due\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})', re.IGNORECASE),
            re.compile(r'due\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})', re.IGNORECASE),
        ]
        
        for pattern in due_patterns:
            match = pattern.search(text)
            if match:
                info['due_date'] = match.group(1).strip()
                break
        
        # Total amount
        total_patterns = [
            re.compile(r'total\s+amount\s*:?\s*([0-9,]+\.?\d*)', re.IGNORECASE),
            re.compile(r'amount\s+due\s*:?\s*([0-9,]+\.?\d*)', re.IGNORECASE),
            re.compile(r'total\s*:?\s*([0-9,]+\.?\d*)', re.IGNORECASE),
        ]
        
        for pattern in total_patterns:
            match = pattern.search(text)
            if match:
                info['total_amount'] = match.group(1).strip()
                break
        
        return info
    
    def _extract_contract_info(self, text):
        """Extract contract-specific information"""
        info = {}
        
        # Contract parties
        party_patterns = [
            re.compile(r'between\s+([^,\n]+)\s+and\s+([^,\n]+)', re.IGNORECASE),
            re.compile(r'party\s+(?:of\s+the\s+)?(?:first\s+part|a)\s*:?\s*([^\n]+)', re.IGNORECASE),
            re.compile(r'party\s+(?:of\s+the\s+)?(?:second\s+part|b)\s*:?\s*([^\n]+)', re.IGNORECASE),
        ]
        
        for pattern in party_patterns:
            match = pattern.search(text)
            if match:
                if 'between' in pattern.pattern.lower():
                    info['party_a'] = match.group(1).strip()
                    info['party_b'] = match.group(2).strip()
                else:
                    key = 'party_a' if 'first' in pattern.pattern or 'party a' in pattern.pattern.lower() else 'party_b'
                    info[key] = match.group(1).strip()
        
        # Contract term/duration
        term_patterns = [
            re.compile(r'term\s+of\s+([^.\n]+)', re.IGNORECASE),
            re.compile(r'duration\s+of\s+([^.\n]+)', re.IGNORECASE),
            re.compile(r'period\s+of\s+([^.\n]+)', re.IGNORECASE),
        ]
        
        for pattern in term_patterns:
            match = pattern.search(text)
            if match:
                info['contract_term'] = match.group(1).strip()
                break
        
        return info
    
    def _extract_email_info(self, text):
        """Extract email-specific information"""
        info = {}
        
        # Email patterns
        patterns = {
            'sender': re.compile(r'from\s*:?\s*([^\n]+)', re.IGNORECASE),
            'recipient': re.compile(r'to\s*:?\s*([^\n]+)', re.IGNORECASE),
            'subject': re.compile(r'subject\s*:?\s*([^\n]+)', re.IGNORECASE),
            'date_sent': re.compile(r'(?:date|sent)\s*:?\s*([^\n]+)', re.IGNORECASE),
            'cc': re.compile(r'cc\s*:?\s*([^\n]+)', re.IGNORECASE),
        }
        
        for key, pattern in patterns.items():
            match = pattern.search(text)
            if match:
                info[key] = match.group(1).strip()
        
        return info
    
    def _extract_financial_info(self, text):
        """Extract financial statement information"""
        info = {}
        
        # Financial terms
        financial_patterns = {
            'net_income': re.compile(r'net\s+income\s*:?\s*([0-9,\.]+)', re.IGNORECASE),
            'total_revenue': re.compile(r'(?:total\s+)?revenue\s*:?\s*([0-9,\.]+)', re.IGNORECASE),
            'total_assets': re.compile(r'total\s+assets\s*:?\s*([0-9,\.]+)', re.IGNORECASE),
            'gross_profit': re.compile(r'gross\s+profit\s*:?\s*([0-9,\.]+)', re.IGNORECASE),
            'operating_income': re.compile(r'operating\s+income\s*:?\s*([0-9,\.]+)', re.IGNORECASE),
        }
        
        for key, pattern in financial_patterns.items():
            match = pattern.search(text)
            if match:
                info[key] = match.group(1).strip()
        
        return info
    
    def _extract_report_info(self, text):
        """Extract report-specific information"""
        info = {}
        
        # Report sections
        if 'executive summary' in text.lower():
            info['has_executive_summary'] = True
        
        if 'recommendations' in text.lower():
            info['has_recommendations'] = True
        
        if 'conclusion' in text.lower():
            info['has_conclusion'] = True
        
        # Extract key metrics mentioned
        metrics = re.findall(r'(\d+(?:\.\d+)?%)', text)
        if metrics:
            info['percentages_mentioned'] = metrics[:5]  # Limit to 5
        
        # Report period
        period_patterns = [
            re.compile(r'(?:for\s+the\s+)?(?:period|quarter|year)\s+(?:ended|ending)\s+([^\n,]+)', re.IGNORECASE),
            re.compile(r'(?:monthly|quarterly|annual)\s+report\s+(?:for\s+)?([^\n,]+)', re.IGNORECASE),
        ]
        
        for pattern in period_patterns:
            match = pattern.search(text)
            if match:
                info['report_period'] = match.group(1).strip()
                break
        
        return info
    
    def _extract_receipt_info(self, text):
        """Extract receipt-specific information"""
        info = {}
        
        # OR number
        or_patterns = [
            re.compile(r'(?:or|official\s+receipt)\s+(?:no|number)\s*:?\s*([A-Z0-9\-]+)', re.IGNORECASE),
            re.compile(r'receipt\s+#?\s*:?\s*([A-Z0-9\-]+)', re.IGNORECASE),
        ]
        
        for pattern in or_patterns:
            match = pattern.search(text)
            if match:
                info['receipt_number'] = match.group(1).strip()
                break
        
        return info
    
    def _extract_purchase_order_info(self, text):
        """Extract purchase order information"""
        info = {}
        
        # PO number
        po_patterns = [
            re.compile(r'(?:po|purchase\s+order)\s+(?:no|number)\s*:?\s*([A-Z0-9\-]+)', re.IGNORECASE),
            re.compile(r'p\.?o\.?\s*#?\s*:?\s*([A-Z0-9\-]+)', re.IGNORECASE),
        ]
        
        for pattern in po_patterns:
            match = pattern.search(text)
            if match:
                info['po_number'] = match.group(1).strip()
                break
        
        return info
    
    def _extract_delivery_receipt_info(self, text):
        """Extract delivery receipt information"""
        info = {}
        
        # DR number
        dr_patterns = [
            re.compile(r'(?:dr|delivery\s+receipt)\s+(?:no|number)\s*:?\s*([A-Z0-9\-]+)', re.IGNORECASE),
            re.compile(r'd\.?r\.?\s*#?\s*:?\s*([A-Z0-9\-]+)', re.IGNORECASE),
        ]
        
        for pattern in dr_patterns:
            match = pattern.search(text)
            if match:
                info['dr_number'] = match.group(1).strip()
                break
        
        # Delivered to
        delivered_to_pattern = re.compile(r'delivered\s+to\s*:?\s*([^\n]+)', re.IGNORECASE)
        match = delivered_to_pattern.search(text)
        if match:
            info['delivered_to'] = match.group(1).strip()
        
        return info
    
    def _extract_payroll_info(self, text):
        """Extract payroll information"""
        info = {}
        
        # Employee ID
        emp_id_patterns = [
            re.compile(r'(?:employee|emp)\s+(?:id|number)\s*:?\s*([A-Z0-9\-]+)', re.IGNORECASE),
            re.compile(r'id\s+(?:no|number)\s*:?\s*([A-Z0-9\-]+)', re.IGNORECASE),
        ]
        
        for pattern in emp_id_patterns:
            match = pattern.search(text)
            if match:
                info['employee_id'] = match.group(1).strip()
                break
        
        # Pay period
        period_patterns = [
            re.compile(r'pay\s+period\s*:?\s*([^\n]+)', re.IGNORECASE),
            re.compile(r'period\s+covered\s*:?\s*([^\n]+)', re.IGNORECASE),
        ]
        
        for pattern in period_patterns:
            match = pattern.search(text)
            if match:
                info['pay_period'] = match.group(1).strip()
                break
        
        return info
    
    def _extract_generic_info(self, text, entities):
        """Extract generic key information for unknown document types"""
        info = {}
        
        # Reference numbers - more comprehensive
        ref_patterns = [
            re.compile(r'(?:ref|reference)\s*#?\s*:?\s*([A-Z0-9\-]+)', re.IGNORECASE),
            re.compile(r'(?:no|number)\s*\.?\s*:?\s*([A-Z0-9\-]+)', re.IGNORECASE),
            re.compile(r'document\s+(?:no|number)\s*:?\s*([A-Z0-9\-]+)', re.IGNORECASE),
            re.compile(r'control\s+(?:no|number)\s*:?\s*([A-Z0-9\-]+)', re.IGNORECASE),
        ]
        
        for pattern in ref_patterns:
            matches = pattern.findall(text)
            if matches:
                info['reference_numbers'] = list(set(matches))[:5]  # Remove duplicates, limit to 5
                break
        
        # Important numbers that might be IDs, amounts, etc.
        important_numbers = []
        number_patterns = [
            re.compile(r'\b(\d{4,})\b'),  # Numbers with 4+ digits
            re.compile(r'\b(\d{1,3}(?:,\d{3})+)\b'),  # Comma-separated numbers
        ]
        
        for pattern in number_patterns:
            matches = pattern.findall(text)
            important_numbers.extend(matches)
        
        if important_numbers:
            info['important_numbers'] = list(set(important_numbers))[:10]  # Remove duplicates, limit to 10
        
        # Phone numbers
        phone_patterns = [
            re.compile(r'\b(\+?63\s?9\d{2}\s?\d{3}\s?\d{4})\b'),  # Philippine mobile
            re.compile(r'\b(\(\d{3}\)\s?\d{3}-\d{4})\b'),  # US format
            re.compile(r'\b(\d{3}-\d{3}-\d{4})\b'),  # US format 2
            re.compile(r'\b(\+\d{1,3}\s?\d{3,4}\s?\d{3,4}\s?\d{3,4})\b'),  # International
        ]
        
        phones = []
        for pattern in phone_patterns:
            phones.extend(pattern.findall(text))
        
        if phones:
            info['phone_numbers'] = list(set(phones))[:5]
        
        # Email addresses
        email_pattern = re.compile(r'\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b')
        emails = email_pattern.findall(text)
        if emails:
            info['email_addresses'] = list(set(emails))[:5]
        
        # Addresses - basic extraction
        address_patterns = [
            re.compile(r'\b(\d+\s+[A-Za-z\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Circle|Cir|Plaza|Plz)(?:\s*,\s*[A-Za-z\s]+)*)\b', re.IGNORECASE),
        ]
        
        addresses = []
        for pattern in address_patterns:
            addresses.extend(pattern.findall(text))
        
        if addresses:
            info['addresses'] = list(set(addresses))[:3]
        
        # Extract potential plate numbers if vehicle-related
        if any(word in text.lower() for word in ['vehicle', 'car', 'truck', 'motorcycle', 'plate', 'registration']):
            plate_pattern = re.compile(r'\b([A-Z]{2,3}\s*\d{3,4}|[A-Z]{3}\s*\d{3}|[A-Z0-9]{6,8})\b')
            plates = plate_pattern.findall(text)
            if plates:
                info['plate_numbers'] = list(set(plates))[:5]
        
        return info
    
    def _calculate_confidence(self, ocr_confidence, extraction_success, entities, key_info):
        """Calculate processing confidence based on multiple factors"""
        base_confidence = ocr_confidence
        
        # Boost for successful entity extraction
        entity_count = sum(len(entities[key]) for key in entities)
        entity_bonus = min(entity_count * 3, 20)  # Max 20 points
        
        # Boost for key information extracted
        key_info_bonus = min(len(key_info) * 2, 15)  # Max 15 points
        
        # Extraction success bonus
        extraction_bonus = 10 if extraction_success else 0
        
        # Quality bonus based on entity diversity
        diversity_bonus = 0
        if entities['people'] and entities['companies']:
            diversity_bonus += 5
        if entities['locations']:
            diversity_bonus += 3
        
        total_confidence = base_confidence + entity_bonus + key_info_bonus + extraction_bonus + diversity_bonus
        return min(total_confidence, 100.0)
    
    def process_single_file(self, file_path, category_name, filename):
        """Process a single file with dynamic extraction"""
        start_time = time.time()
        
        try:
            # Step 1: OCR extraction
            ocr_result = self._ocr_processor.extract_text_from_pdf(file_path)
            ocr_text = ocr_result.get("text_only", "")
            ocr_confidence = ocr_result.get("avg_confidence", 0.0)
            
            if not ocr_text or len(ocr_text.strip()) < 20:
                raise Exception("Insufficient OCR text extracted")
            
            # Step 2: Determine document type
            document_type = self._determine_document_type(ocr_text, filename)
            
            # Step 3: Extract entities
            entities = self._extract_entities(ocr_text)
            
            # Step 4: Extract dates and amounts
            dates, amounts = self._extract_dates_and_amounts(ocr_text)
            
            # Step 5: Extract key information based on document type
            key_information = self._extract_key_information(ocr_text, document_type, entities)
            
            # Step 6: Try AI-powered extraction for additional insights
            extraction_success = False
            ai_confidence = 0.0
            extraction_method = "rule_based"
            
            try:
                ai_result = self._data_extractor.extract_structured_data(ocr_text, document_type)
                if ai_result and ai_result.get('data'):
                    # Merge AI results with rule-based extraction
                    ai_data = ai_result.get('data', {})
                    key_information.update(ai_data)
                    ai_confidence = ai_result.get('confidence', 0.0) * 100
                    extraction_success = True
                    extraction_method = "ai_enhanced"
            except Exception as e:
                logger.warning(f"AI extraction failed for {filename}: {e}")
            
            # Step 7: Calculate overall confidence
            processing_confidence = self._calculate_confidence(
                ocr_confidence, extraction_success, entities, key_information
            )
            
            # Step 8: Create dynamic metadata
            metadata = {
                "document_type": document_type,
                "confidence": round(ai_confidence / 100, 2) if ai_confidence else 0.75,
                "entities": entities,
                "key_information": key_information,
                "dates_found": dates,
                "amounts_found": amounts,
                "processing_confidence": round(processing_confidence, 1),
                "extraction_method": extraction_method,
                "processing_info": {
                    "category": category_name,
                    "classification": "successful" if processing_confidence >= 70 else "partial" if processing_confidence >= 40 else "failed",
                    "document_name": self.archive_manager.generate_document_name(filename, category_name),
                    "original_filename": filename,
                    "processed_at": timezone.now().astimezone(pytz.timezone('Asia/Manila')).isoformat(),
                    "ocr_confidence": round(ocr_confidence, 1),
                    "text_length": len(ocr_text),
                    "entity_count": sum(len(entities[key]) for key in entities),
                    "key_info_fields": len(key_information)
                }
            }
            
            # Step 9: Archive if successful
            classification = metadata["processing_info"]["classification"]
            if classification in ['successful', 'partial']:
                archive_paths = self.archive_manager.create_archive_structure(
                    category_name=category_name,
                    document_name=metadata["processing_info"]["document_name"],
                    pdf_path=file_path,
                    ocr_text=ocr_text,
                    metadata=metadata,
                    classification=classification
                )
                
                self.archive_manager.cleanup_upload_file(file_path)
                processing_time = time.time() - start_time
                
                # Update stats
                self.stats['total_processed'] += 1
                self.stats['successful'] += 1
                self.stats['total_time'] += processing_time
                
                if document_type not in self.stats['document_types']:
                    self.stats['document_types'][document_type] = 0
                self.stats['document_types'][document_type] += 1
                
                logger.info(f"SUCCESS: {filename} ({document_type}) - {processing_confidence:.1f}% confidence, "
                           f"{sum(len(entities[key]) for key in entities)} entities, "
                           f"{len(key_information)} key fields in {processing_time:.1f}s")
                
                return {
                    'status': 'completed',
                    'classification': classification,
                    'document_type': document_type,
                    'confidence': processing_confidence,
                    'processing_time': processing_time,
                    'entities_found': sum(len(entities[key]) for key in entities),
                    'key_info_fields': len(key_information)
                }
            else:
                processing_time = time.time() - start_time
                self.stats['total_processed'] += 1
                self.stats['failed'] += 1
                self.stats['total_time'] += processing_time
                
                logger.warning(f"LOW CONFIDENCE: {filename} - {processing_confidence:.1f}% confidence")
                return {
                    'status': 'skipped',
                    'classification': 'failed',
                    'confidence': processing_confidence,
                    'processing_time': processing_time,
                    'reason': 'Low processing confidence'
                }
        
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats['total_processed'] += 1
            self.stats['failed'] += 1
            self.stats['total_time'] += processing_time
            
            logger.error(f"ERROR: {filename} - {str(e)}")
            return {
                'status': 'skipped',
                'classification': 'failed',
                'processing_time': processing_time,
                'error': str(e)
            }
    
    def process_category_parallel(self, category_name):
        """Process category with parallel execution"""
        logger.info(f"Processing category: {category_name}")
        
        categories_data = self.archive_manager.scan_uploads_categories()
        category_data = next((c for c in categories_data if c['category_name'] == category_name), None)
        
        if not category_data or not category_data.get('pdf_files'):
            return {'total': 0, 'successful': 0, 'failed': 0, 'skipped': 0}
        
        pdf_files = category_data['pdf_files']
        results = {'total': len(pdf_files), 'successful': 0, 'failed': 0, 'skipped': 0}
        
        logger.info(f"Processing {len(pdf_files)} files with {self.max_workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_single_file, pdf['full_path'], category_name, pdf['filename']): pdf
                for pdf in pdf_files
            }
            
            for future in concurrent.futures.as_completed(future_to_file, timeout=3600):
                pdf_info = future_to_file[future]
                
                try:
                    result = future.result(timeout=600)  # 10 min timeout per file
                    
                    if result:
                        if result.get('status') == 'completed':
                            results['successful'] += 1
                        else:
                            results['failed'] += 1
                    else:
                        results['failed'] += 1
                        
                except Exception as e:
                    logger.error(f"Processing failed for {pdf_info['filename']}: {e}")
                    results['failed'] += 1
        
        # Cleanup memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Category {category_name} completed: {results}")
        return results
    
    def process_all_categories(self):
        """Process all categories"""
        categories_data = self.archive_manager.scan_uploads_categories()
        all_results = []
        
        for category_data in categories_data:
            category_name = category_data['category_name']
            file_count = len(category_data.get('pdf_files', []))
            
            if file_count == 0:
                continue
            
            logger.info(f"Starting category: {category_name} ({file_count} files)")
            results = self.process_category_parallel(category_name)
            all_results.append({'name': category_name, 'results': results})
        
        return all_results
    
    def get_stats_summary(self):
        """Get processing statistics summary"""
        if self.stats['total_processed'] == 0:
            return {
                'total_processed': 0,
                'success_rate': 0.0,
                'failed_count': 0,
                'average_time': 0.0,
                'document_types': {}
            }
        
        avg_time = self.stats['total_time'] / self.stats['total_processed']
        success_rate = (self.stats['successful'] / self.stats['total_processed']) * 100
        
        return {
            'total_processed': self.stats['total_processed'],
            'success_rate': round(success_rate, 1),
            'failed_count': self.stats['failed'],
            'average_time': round(avg_time, 2),
            'throughput_per_hour': round(3600 / avg_time) if avg_time > 0 else 0,
            'document_types': self.stats['document_types'],
            'processing_summary': {
                'total_entities_extracted': sum(len(self.stats.get('entities', {}).get(key, [])) for key in ['people', 'companies', 'locations']),
                'average_confidence': round(sum(self.stats.get('confidences', [85])) / len(self.stats.get('confidences', [1])), 1) if self.stats.get('confidences') else 85.0
            }
        }


class Command(BaseCommand):
    """Django management command for document processing"""
    
    help = 'Process uploaded documents with enhanced pattern recognition'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--category',
            type=str,
            help='Process specific category only',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be processed without actually processing',
        )
        parser.add_argument(
            '--force-cpu',
            action='store_true',
            help='Force CPU-only processing',
        )
        parser.add_argument(
            '--max-workers',
            type=int,
            default=None,
            help='Maximum number of worker threads',
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose logging',
        )
    
    def handle(self, *args, **options):
        if options['force_cpu']:
            os.environ['FORCE_CPU_ONLY'] = '1'
        
        if options['verbose']:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Initialize processor
        processor = DocumentProcessor()
        
        if options['max_workers']:
            processor.max_workers = min(options['max_workers'], cpu_count())
        
        start_time = time.time()
        
        try:
            if options['dry_run']:
                self.stdout.write(self.style.WARNING('DRY RUN MODE - No files will be processed'))
                categories_data = processor.archive_manager.scan_uploads_categories()
                
                for category_data in categories_data:
                    category_name = category_data['category_name']
                    file_count = len(category_data.get('pdf_files', []))
                    
                    if options['category'] and category_name != options['category']:
                        continue
                    
                    self.stdout.write(f"Category: {category_name} - {file_count} files")
                    
                    for pdf in category_data.get('pdf_files', [])[:5]:  # Show first 5
                        self.stdout.write(f"  - {pdf['filename']}")
                    
                    if file_count > 5:
                        self.stdout.write(f"  ... and {file_count - 5} more files")
                
                return
            
            if options['category']:
                # Process specific category
                self.stdout.write(f"Processing category: {options['category']}")
                results = processor.process_category_parallel(options['category'])
                
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Category '{options['category']}' completed: "
                        f"{results['successful']} successful, {results['failed']} failed"
                    )
                )
            else:
                # Process all categories
                self.stdout.write("Processing all categories...")
                all_results = processor.process_all_categories()
                
                # Display results
                total_successful = sum(r['results']['successful'] for r in all_results)
                total_failed = sum(r['results']['failed'] for r in all_results)
                
                self.stdout.write("\nProcessing Summary:")
                for result in all_results:
                    category = result['name']
                    stats = result['results']
                    self.stdout.write(
                        f"  {category}: {stats['successful']} successful, "
                        f"{stats['failed']} failed of {stats['total']} total"
                    )
                
                self.stdout.write(
                    self.style.SUCCESS(
                        f"\nOverall: {total_successful} successful, {total_failed} failed"
                    )
                )
            
            # Display final statistics
            stats = processor.get_stats_summary()
            total_time = time.time() - start_time
            
            self.stdout.write(f"\nFinal Statistics:")
            self.stdout.write(f"  Total processed: {stats['total_processed']}")
            self.stdout.write(f"  Success rate: {stats['success_rate']}%")
            self.stdout.write(f"  Average processing time: {stats['average_time']}s per file")
            self.stdout.write(f"  Throughput: {stats['throughput_per_hour']} files/hour")
            self.stdout.write(f"  Total execution time: {total_time:.1f}s")
            
            if stats['document_types']:
                self.stdout.write(f"\nDocument Types Processed:")
                for doc_type, count in sorted(stats['document_types'].items()):
                    self.stdout.write(f"  {doc_type}: {count}")
            
            self.stdout.write(
                self.style.SUCCESS('Document processing completed successfully!')
            )
        
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING('\nProcessing interrupted by user'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Processing failed: {str(e)}'))
            raise