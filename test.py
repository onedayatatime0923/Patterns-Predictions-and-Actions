import nflgame

games = nflgame.games(2019, week=1)
for g in games:
    print(g)
    #  print(g.__dict__)
    players = nflgame.combine_game_stats([g])
    for p in players.rushing().sort('rushing_yds'):
        msg = '%s %d carries for %d yards and %d TDs'
        print(msg % (p, p.rushing_att, p.rushing_yds, p.rushing_tds))
    input()
    #  print(players)

#  players = nflgame.combine_game_stats(games)
#  for p in players.rushing().sort('rushing_yds').limit(5):
#      msg = '%s %d carries for %d yards and %d TDs'
#      print(msg % (p, p.rushing_att, p.rushing_yds, p.rushing_tds))
