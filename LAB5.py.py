# Lab 5 - the "what to wear" bot.
#
# what this does: the user types a city, and the bot tells them what to wear today
# and what outdoor activities suit the weather. it is NOT a chatbot. there is no back
# and forth, just one input (a city) and one output (the advice).
#
# how it works (this is the point of the lab - openai "tool calling"):
#   step 1: i give the model a weather TOOL and ask my question. with tool_choice="auto"
#           the model decides on its own that it needs the weather, and replies asking me
#           to run get_current_weather for a specific city. it does not run anything itself.
#   step 2: MY code runs get_current_weather() and gets the real weather data.
#   step 3: i hand that weather back to the model and make a SECOND call. now the model
#           has the numbers it needs, so it writes the actual clothing + activity advice.
# so it takes two calls to the api, not one.

import json
import requests
import streamlit as st
from openai import OpenAI

# same model i used in my other labs
CHAT_MODEL = "gpt-5-mini"

st.title("Lab 5 - What to Wear Bot")
st.write("Type a city and i'll tell you what to wear today and what to do outside.")

# read the api key from secrets, same as my other labs.
# locally this comes from .streamlit/secrets.toml, on streamlit cloud it comes from
# the app's Secrets settings. wttr.in needs no key, so openai is the only one.
if "OPENAI_API_KEY" not in st.secrets:
    st.error("No OPENAI_API_KEY found in secrets. Add it to .streamlit/secrets.toml "
             "(and to your app's Secrets on Streamlit Cloud).")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ------------------ Part A: the weather function ------------------
# get today's weather for a location from wttr.in.
# location can be a city, a zip code, an airport code like 'SYR', or a landmark.
# wttr.in is free and needs no api key. adding ?format=j1 makes it return json.
#
# i decided which values the bot actually needs to give good advice. the starter only
# returned temp and description, but to dress for a whole day you need to know how the
# day changes. so i pull: current temp and feels-like, wind, humidity, today's high and
# low, sunrise and sunset, the day's peak rain and snow chance, and an hourly outlook.
def get_current_weather(location):
    # if no location was given, default to Syracuse (matches step 6b)
    if not location:
        location = "Syracuse, NY"

    # build the url and ask wttr.in for json
    url = f"https://wttr.in/{location}?format=j1"
    response = requests.get(url, timeout=10)

    # anything other than 200 means the request failed
    if response.status_code != 200:
        raise Exception(f"wttr.in error: status {response.status_code}")

    # unknown places come back as plain text instead of json, so .json() will fail.
    # if that happens, treat it as "location not found".
    try:
        data = response.json()
    except ValueError:
        raise Exception(f"Could not find a location named {location}")

    # j1 has three sections i care about:
    #   current_condition -> conditions right now
    #   weather           -> one entry per day (i only use today, index 0)
    #   nearest_area      -> the place wttr.in actually matched my text to
    current = data["current_condition"][0]
    today = data["weather"][0]
    astronomy = today["astronomy"][0]

    # today["hourly"] is the forecast in 3-hour steps. i scan it for the day's
    # highest chance of rain and snow, so the bot knows if weather is coming later.
    max_rain = max(int(h["chanceofrain"]) for h in today["hourly"])
    max_snow = max(int(h["chanceofsnow"]) for h in today["hourly"])

    # build a small hour-by-hour list so the bot can see how the day changes.
    # wttr gives the time as 0, 300, 600 ... 2100 (basically HHMM with no colon),
    # so dividing by 100 gives the hour. e.g. 1500 -> 15 -> "15:00".
    hourly = []
    for h in today["hourly"]:
        hhmm = int(h["time"])
        hourly.append({
            "time": f"{hhmm // 100:02d}:00",
            "temp_F": int(h["tempF"]),
            "feels_like_F": int(h["FeelsLikeF"]),
            "chance_of_rain": int(h["chanceofrain"]),
            "description": h["weatherDesc"][0]["value"],
        })

    # build a readable name of the place wttr.in matched, e.g. "Syracuse, New York, United States".
    # each of these is a list with the value nested one level down, so i pull part[0]["value"].
    area = data["nearest_area"][0]
    matched = ", ".join(
        part[0]["value"]
        for part in (area.get("areaName"), area.get("region"), area.get("country"))
        if part and part[0]["value"]
    )

    # return everything as a clean dictionary. this is what gets handed to the model.
    return {
        "location": matched or location,
        "current_temp_F": int(float(current["temp_F"])),
        "feels_like_F": int(float(current["FeelsLikeF"])),
        "description": current["weatherDesc"][0]["value"],
        "wind_mph": int(current["windspeedMiles"]),
        "humidity_pct": int(current["humidity"]),
        "today_high_F": int(today["maxtempF"]),
        "today_low_F": int(today["mintempF"]),
        "max_chance_of_rain": max_rain,
        "max_chance_of_snow": max_snow,
        "sunrise": astronomy["sunrise"],
        "sunset": astronomy["sunset"],
        "hourly": hourly,
    }


# ------------------ Part B: describe the tool to the model ------------------
# this is how the model "knows" my function exists. i am not giving it the code,
# just a description: the name, what it does, and what arguments it takes. the model
# reads this and, when it thinks it needs weather, replies asking me to run it with a
# location. the "location" argument is what it fills in from the user's city.
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": (
                "Get today's weather for a city: current conditions, today's high/low, "
                "sunrise/sunset, chance of rain or snow, and an hourly outlook."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name like 'Syracuse, NY' or 'Paris'. "
                                       "Can also be a zip code or airport code.",
                    }
                },
                "required": ["location"],
            },
        },
    }
]


# ------------------ the bot: input a city, output advice ------------------
# a plain text box and a button. this is what keeps it a one-shot tool and not a chatbot.
city = st.text_input("Enter a city", placeholder="Syracuse, NY")

# everything below only runs when the button is clicked
if st.button("What should I wear today?"):
    # if the box was left empty, default to Syracuse (step 6b)
    location = city.strip() or "Syracuse, NY"

    # the system prompt tells the model its job: use the tool, then give clothing +
    # activity advice, and think about how the day changes (morning vs afternoon vs evening).
    system_prompt = (
        "You are a friendly 'what to wear' assistant. The user gives you a city. "
        "Use the get_current_weather tool to look up today's weather for that city, then "
        "recommend what clothes to wear today and suggest outdoor activities that suit the "
        "weather. Consider how conditions change through the day (morning, afternoon, evening), "
        "the temperature and feels-like, wind, and the chance of rain or snow. "
        "Keep it practical and easy to skim."
    )

    # start the conversation: my instructions, then the user's request built from the city
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": f"What should I wear in {location} today, and what can I do outside?"},
    ]

    # FIRST call: send the question AND the tool. tool_choice="auto" (step 6a) lets the
    # model decide whether to use the tool. for a "what to wear" question it always will.
    first = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    msg = first.choices[0].message

    # did the model ask to run my tool? if yes, this is the whole point of the lab.
    if msg.tool_calls:
        # add the model's tool request to the conversation so the history stays valid
        messages.append(msg)

        # run the tool for each call the model made (normally just one)
        weather = None
        for tool_call in msg.tool_calls:
            # the model sends the arguments as a json string, so i parse it back to a dict
            args = json.loads(tool_call.function.arguments)
            loc = args.get("location") or "Syracuse, NY"   # default again if it came back empty

            # actually run my function. if the city is bad, catch it so the app doesn't crash.
            try:
                weather = get_current_weather(loc)
            except Exception as e:
                weather = {"error": str(e)}

            # hand the result back to the model as a "tool" message, tied to this call's id
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(weather),
            })

        # show the user the weather i pulled (nice to see, not required)
        if weather and "error" not in weather:
            st.info(
                f"**{weather['location']}** - {weather['description']}, "
                f"now {weather['current_temp_F']}F (feels {weather['feels_like_F']}F). "
                f"High {weather['today_high_F']}F / Low {weather['today_low_F']}F. "
                f"Rain chance up to {weather['max_chance_of_rain']}%, "
                f"wind {weather['wind_mph']} mph."
            )

        # SECOND call (step 7): the weather is now in the messages, so i ask the model to
        # write the advice. i stream it so it appears word by word like my other labs.
        st.subheader("Today's recommendation")
        second = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            stream=True,
        )
        st.write_stream(second)
    else:
        # the model decided it did not need the weather. just show whatever it said.
        st.write(msg.content)
