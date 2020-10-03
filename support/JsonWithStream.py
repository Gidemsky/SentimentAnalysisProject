import ijson

from support.Utils import get_json_tweet_list, create_json_dict_file

JFILE = r"C:\Users\dembo\Documents\Computer Science\Third Year\Project\Sentiment Analysis Project\Models\Data\Train-Set.json"

dataset = ijson.parse(open(JFILE, encoding="utf8"))

data_set_try = list()

is_reading_tweet = False
i = 0

tweet_dict = {}

text_dict = {}

k = None
sub_k = None
sub_sub_k = None
v = None
sub_v = None
sub_sub_v = None

start_list = False

for prefix, type_of_object, value in dataset:
    print(prefix, type_of_object, value)
    if prefix == 'tweets.item' and type_of_object == 'start_map':
        is_reading_tweet = True
    elif prefix == 'tweets.item' and type_of_object == 'end_map':
        i += 1
        is_reading_tweet = False
        data_set_try.append(tweet_dict)
        tweet_dict = {}

    if is_reading_tweet:
        if prefix == 'tweets.item' and type_of_object == 'map_key' and value == 'id':
            k = value
        elif prefix == 'tweets.item.id' and type_of_object == 'number':
            v = value
            tweet_dict[k] = v

        elif prefix == 'tweets.item' and type_of_object == 'map_key' and value == 'id_str':
            k = value
        elif prefix == 'tweets.item.id_str' and type_of_object == 'string':
            v = value
            tweet_dict[k] = v

        elif prefix == 'tweets.item' and type_of_object == 'map_key' and value == 'text':
            text_dict = {}
            k = value
        elif prefix == 'tweets.item.text.item' and type_of_object == 'map_key':
            sub_k = value
        elif prefix == 'tweets.item.text.item.translatedText' and type_of_object == 'string':
            sub_v = value
            text_dict[sub_k] = sub_v
        elif prefix == 'tweets.item.text.item.input' and type_of_object == 'string':
            sub_v = value
            text_dict[sub_k] = sub_v
        elif prefix == 'tweets.item.text' and type_of_object == 'end_array':
            v = list()
            v.append(text_dict)
            tweet_dict[k] = v

        elif prefix == 'tweets.item' and type_of_object == 'map_key' and value == 'extended_tweet':
            text_dict = {}
            k = value
        elif prefix == 'tweets.item.extended_tweet.full_text.item' and type_of_object == 'map_key':
            sub_k = value
        elif prefix == 'tweets.item.extended_tweet.full_text.item.translatedText' and type_of_object == 'string':
            sub_v = value
            text_dict[sub_k] = sub_v
        elif prefix == 'tweets.item.extended_tweet.full_text.item.input' and type_of_object == 'string':
            sub_v = value
            text_dict[sub_k] = sub_v
        elif prefix == 'tweets.item.extended_tweet.full_text' and type_of_object == 'end_array':
            v = list()
            v.append(text_dict)
            tweet_dict[k] = {"full_text": v}

        elif prefix == 'tweets.item' and type_of_object == 'map_key' and value == 'label':
            text_dict = {}
            k = value
        elif prefix == 'tweets.item.label' and type_of_object == 'map_key':
            sub_k = value
        elif prefix == 'tweets.item.label.positivity' and type_of_object == 'string':
            sub_v = value
            text_dict[sub_k] = sub_v
        elif prefix == 'tweets.item.label.relative subject' and type_of_object == 'string':
            if value == "0":
                value = "person"
            elif value == "1":
                value = "topic"
            sub_v = value
            text_dict[sub_k] = sub_v
        elif prefix == 'tweets.item.label' and type_of_object == 'end_map':
            tweet_dict[k] = text_dict

create_json_dict_file(data_set_try, "Train-Set2")
print("number of tweets: " + str(i))
