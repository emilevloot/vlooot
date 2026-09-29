# Encoding: exactly the numbers of mp_encode.encode_reference
# ---------------------------------------------------------------------------

@njit(cache=True)
def _chains(s, gs, p, comp, size, adj):
    """Label the chains of player p's Vikings: comp[i] = chain number of
    board space i (-1: none of p's Vikings), size[k] = its number of spaces,
    adj[k] = bitmask (bit = board space) of the watchtowers and castles next
    to it (two int64 words: spaces 0-63 and 64-127). Returns the number of chains."""
    for i in range(NL):
        comp[i] = -1
    todo = np.empty(NB, np.int64)
    nc = 0
    for start in range(NB):
        if comp[start] >= 0 or s[S_VIK + p * NL + start] == 0:
            continue
        comp[start] = nc
        todo[0] = start
        n = 1
        head = 0
        m0 = np.int64(0)
        m1 = np.int64(0)
        while head < n:
            c = todo[head]
            head += 1
            for d in range(6):
                nb = LAND_NB[c, d]
                if nb >= NB:
                    continue
                t = gs[G_TERR + nb]
                if t == T_TOWER or t == T_CASTLE:
                    if nb < 64:
                        m0 |= np.int64(1) << nb
                    else:
                        m1 |= np.int64(1) << (nb - 64)
                if comp[nb] < 0 and s[S_VIK + p * NL + nb] > 0:
                    comp[nb] = nc
                    todo[n] = nb
                    n += 1
        size[nc] = n
        adj[nc, 0] = m0
        adj[nc, 1] = m1
        nc += 1
    return nc


@njit(cache=True)
def _bit(m0, m1, a):
    if a < 64:
        return (m0 >> a) & 1
    return (m1 >> (a - 64)) & 1


@njit(cache=True)
def _placement_gain(s, gs, p, c, comp, size, adj, need, out):
    """What player p would get from a Viking on free space c (no shields),
    without changing the game: out = [chain size, house, watchtower and
    castle tiles, useful items] (see nn_encode._placement_gain)."""
    seen = np.full(6, -1, np.int64)
    ns = 0
    total = 1
    m0 = np.int64(0)
    m1 = np.int64(0)
    for d in range(6):
        nb = LAND_NB[c, d]
        if nb >= NB:
            continue
        t = gs[G_TERR + nb]
        if t == T_TOWER or t == T_CASTLE:
            if nb < 64:
                m0 |= np.int64(1) << nb
            else:
                m1 |= np.int64(1) << (nb - 64)
        k = comp[nb]
        if k >= 0:
            new = True
            for j in range(ns):
                if seen[j] == k:
                    new = False
            if new:
                seen[ns] = k
                ns += 1
                total += size[k]
                m0 |= adj[k, 0]
                m1 |= adj[k, 1]
    houses = 0
    towers_got = 0
    castles = 0
    if gs[G_RULES + R_BUILD] > 0:
        for d in range(6):
            nb = LAND_NB[c, d]
            if nb < NB and gs[G_TERR + nb] == T_HOUSE and s[S_STOCK + nb] > 0:
                houses += 1
        towers = np.empty(MAXT, np.int64)
        m = 0
        for j in range(NB):
            a = COORD_ORDER[j]
            if _bit(m0, m1, a) and gs[G_TERR + a] == T_TOWER:
                towers[m] = a
                m += 1
        grp = np.full(m, -1, np.int64)
        ng = tower_groups(s, gs, p, towers, m, grp)
        if ng >= 2:
            pay = np.empty(MAXT, np.int64)
            tower_choice(s, towers, m, grp, ng, c, pay)
            for j in range(ng):
                if s[S_STOCK + pay[j]] > 0:
                    towers_got += 1
        level = min(3, total // 4)
        for a in range(NB):
            if _bit(m0, m1, a) and gs[G_TERR + a] == T_CASTLE:
                have = s[S_CASTLE + p * NL + a]
                if level > have:
                    castles += min(level - have, s[S_STOCK + a])
    out[0] = total
    out[1] = houses
    out[2] = towers_got
    out[3] = castles
    out[4] = (min(1, need[gs[G_TERR + c]]) + min(houses, need[4])
              + min(towers_got, need[5]) + min(castles, need[6]))


@njit(cache=True)
def _encode_fjord(s, gs, p, out, need):
    """nn_encode's fjord numbers for player p into out[NF, FJORD_F] (all but
    the clock), and the items still missing on ships/sites that can be
    finished into need[7]. Returns the number of empty spaces where a new
    longship could still be filled."""
    gain = s[S_VLEFT + p] + (1 if s[S_SHIELD + p * 3 + 2] > 0 else 0)
    cnt = np.zeros(7, np.int64)
    spots = 0
    for t in range(7):
        need[t] = 0
    for c in range(NF):
        code = s[S_FJ + p * NF + c]
        room = 0
        hres = 0
        for d in range(6):
            n = FJ_NB[c, d]
            if n < NF:
                cn = s[S_FJ + p * NF + n]
                if cn == 0:
                    room += 1
                elif cn <= 4:
                    hres += 1
        out[c, FF_ROOM] = room
        if code == 0:
            out[c, FF_EMPTY] = 1
            h = min(hres, 3)
            if h + room >= 3 and 3 - h <= gain:
                spots += 1
        elif code <= 4:
            out[c, FF_RES + code - 1] = 1
        elif code <= 7:
            out[c, FF_BLD + code - 5] = 1
        elif code == FJ_SHIP:
            sid = s[S_FJSHIP + p * NF + c]
            filled = s[S_FJFLAG + p * NF + c] > 0
            out[c, FF_SHIP_DONE if filled else FF_SHIP] = 1
            out[c, FF_SHIP_BONUS + LS_CAT[sid]] = LS_BONUS[sid]
            if not filled:
                adj_counts(s, p, c, cnt)
                nmiss = 0
                for r in range(4):
                    m = LS_NEED[sid, r] - cnt[r]
                    if m > 0:
                        out[c, FF_SHIP_MISSING + r] = m
                        nmiss += m
                ok = nmiss <= room and nmiss <= gain
                out[c, FF_FILLABLE] = 1 if ok else 0
                out[c, FF_SLACK] = max(-13, min(13, gain - nmiss))
                if ok:
                    for r in range(4):
                        need[r] += out[c, FF_SHIP_MISSING + r]
        else:
            k = SITE_OF_CELL[c]
            done = s[S_FJFLAG + p * NF + c] > 0
            out[c, FF_SITE_DONE if done else FF_SITE] = 1
            out[c, FF_SITE_VP] = SITE_VP[k]
            if not done:
                adj_counts(s, p, c, cnt)
                nmiss = 0
                nres = 0
                for t in range(7):
                    m = gs[G_SITENEED + (p * 3 + k) * 7 + t] - cnt[t]
                    if m > 0:
                        out[c, FF_SITE_MISSING + t] = m
                        nmiss += m
                        if t < 4:
                            nres += m
                ok = nmiss <= room and nres <= gain
                out[c, FF_FILLABLE] = 1 if ok else 0
                out[c, FF_SLACK] = max(-13, min(13, gain - nmiss))
                if ok:
                    for t in range(7):
                        need[t] += out[c, FF_SITE_MISSING + t]
    return spots


@njit(cache=True)
def _glob_player(s, p, glob, w, spots, need, tl):
    """The global numbers of player p: w = 0 as "me", 1 as "opp"."""
    count = np.zeros(6, np.int64)
    value = np.zeros(6, np.int64)
    sites, unfilled, total = score_parts(s, p, count, value)
    glob[GI_SPOTS[w]] = spots
    glob[GI_VIK[w]] = s[S_VLEFT + p]
    for k in range(3):
        glob[GI_SHIELD[w, k]] = 1 if s[S_SHIELD + p * 3 + k] > 0 else 0
    glob[GI_AXES[w]] = axes(s, p)
    for k in range(6):
        glob[GI_COUNT[w, k]] = count[k]
        glob[GI_VALUE[w, k]] = value[k]
    glob[GI_SITES[w]] = sites
    glob[GI_UNFILLED[w]] = unfilled
    glob[GI_TOTAL[w]] = total
    glob[GI_ROOM[w]] = fj_empty_count(s, p)
    for t in range(7):
        glob[GI_NEED[w, t]] = need[t]
    glob[GI_TURNS[w]] = tl
    glob[GI_TURNS_HOT[w, tl]] = 1


@njit(cache=True)
def turns_left(s, p):
    """Turns player p still gets (the +1 Viking shield puts 2 in one turn)."""
    v = s[S_VLEFT + p]
    return v - 1 if (s[S_SHIELD + p * 3] > 0 and v >= 2) else v


@njit(cache=True)
def encode(s, gs, me, land, fjord, glob):
    """Fill the ZEROED arrays land[NL, LAND_F], fjord[MAXPL, NF, FJORD_F] and
    glob[GLOB_F] with the numbers of mp_encode.encode_reference for `me`."""
    npl = gs[G_NP]
    build = gs[G_RULES + R_BUILD] > 0
    seat = np.empty(npl, np.int64)
    for k in range(npl):
        seat[k] = (me + k) % npl
    tl = np.zeros(MAXPL, np.int64)
    for p in range(npl):
        tl[p] = turns_left(s, p)
    # fjords of every seat, shopping lists
    need = np.zeros((MAXPL, 7), np.int64)
    spots = np.zeros(MAXPL, np.int64)
    for k in range(npl):
        p = seat[k]
        spots[p] = _encode_fjord(s, gs, p, fjord[k], need[p])
        other = 0
        for j in range(npl):
            if j != p:
                other = max(other, tl[j])
        for c in range(NF):
            fjord[k, c, FF_TURNS] = tl[p]
            fjord[k, c, FF_TURNS + 1] = other
    opp_tl = 0
    for k in range(1, npl):
        opp_tl = max(opp_tl, tl[seat[k]])

    comp = np.empty((MAXPL, NL), np.int64)
    size = np.zeros((MAXPL, NB), np.int64)
    adj = np.zeros((MAXPL, NB, 2), np.int64)
    for p in range(npl):
        _chains(s, gs, p, comp[p], size[p], adj[p])
    for k in range(5):
        land[NB + k, LF_INPLAY] = 1
        land[NB + k, LF_OCEAN] = 1
        if s[S_OCEAN + k] >= 0:
            land[NB + k, LF_SHIP] = 1
    gain = np.zeros(N_GAIN, np.int64)
    ogain = np.zeros(N_GAIN, np.int64)
    best = np.zeros((2, 2), np.int64)
    for i in range(NB):
        t = gs[G_TERR + i]
        if t == T_OUT:
            continue
        land[i, LF_INPLAY] = 1
        land[i, LF_TERRAIN + t] = 1
        occupied = False
        vo = 0
        for p in range(npl):
            v = s[S_VIK + p * NL + i]
            if v > 0:
                occupied = True
            if p != me:
                vo += v
        land[i, LF_VIK] = s[S_VIK + me * NL + i]
        land[i, LF_VIK + 1] = min(vo, 127)
        co = 0
        for k in range(1, npl):
            p = seat[k]
            if comp[p, i] >= 0:
                co = max(co, size[p, comp[p, i]])
        if comp[me, i] >= 0:
            land[i, LF_CHAIN] = min(size[me, comp[me, i]], 127)
        land[i, LF_CHAIN + 1] = min(co, 127)
        if t == T_TOWER:
            for k in range(npl):
                p = seat[k]
                links = s[S_PAIRS + p * MAXT + gs[G_TSLOT + i]]
                nl = 0
                while links:
                    nl += links & 1
                    links >>= 1
                touch = 0
                for d in range(6):
                    nb = LAND_NB[i, d]
                    if nb < NB and s[S_VIK + p * NL + nb] > 0:
                        touch = 1
                w = 0 if k == 0 else 1
                land[i, LF_TLINKS + w] = max(land[i, LF_TLINKS + w], nl)
                land[i, LF_TTOUCH + w] = max(land[i, LF_TTOUCH + w], touch)
        elif t == T_CASTLE:
            onext = 0
            for k in range(npl):
                p = seat[k]
                have = s[S_CASTLE + p * NL + i]
                nxt = 0
                if build and have < 3 and s[S_STOCK + i] > 0:
                    biggest = 0
                    for d in range(6):
                        nb = LAND_NB[i, d]
                        if nb < NB and comp[p, nb] >= 0:
                            biggest = max(biggest, size[p, comp[p, nb]])
                    nxt = max(1, 4 * (have + 1) - biggest)
                if k == 0:
                    land[i, LF_CLEVEL] = have
                    land[i, LF_CNEXT] = nxt
                else:
                    land[i, LF_CLEVEL + 1] = max(land[i, LF_CLEVEL + 1], have)
                    if nxt > 0 and (onext == 0 or nxt < onext):
                        onext = nxt
            land[i, LF_CNEXT + 1] = onext
        if t >= T_HOUSE:
            land[i, LF_STOCK + t - T_HOUSE] = s[S_STOCK + i]
        elif is_anchor(s, gs, i):
            if occupied:
                land[i, LF_STACK] = 1
            else:
                land[i, LF_FREE] = 1
                glob[GI_ANCHORS[t]] += 1
                for x in range(N_GAIN):
                    ogain[x] = 0
                obest0 = 0
                obest1 = 0
                for k in range(npl):
                    p = seat[k]
                    _placement_gain(s, gs, p, i, comp[p], size[p], adj[p], need[p], gain)
                    g0 = 1 + gain[1] + gain[2] + gain[3]
                    if k == 0:
                        for x in range(N_GAIN):
                            land[i, LF_GAIN + x] = min(gain[x], 127)
                        best[0, 0] = max(best[0, 0], g0)
                        best[0, 1] = max(best[0, 1], gain[4])
                    else:
                        for x in range(N_GAIN):
                            ogain[x] = max(ogain[x], gain[x])
                        obest0 = max(obest0, g0)
                        obest1 = max(obest1, gain[4])
                for x in range(N_GAIN):
                    land[i, LF_GAIN + N_GAIN + x] = min(ogain[x], 127)
                best[1, 0] = max(best[1, 0], obest0)
                best[1, 1] = max(best[1, 1], obest1)
    for i in range(NL):
        land[i, LF_TURNS] = tl[me]
        land[i, LF_TURNS + 1] = opp_tl

    # global: me and the strongest opponent
    top = seat[1]
    top_s = total_score(s, top)
    for k in range(2, npl):
        sc = total_score(s, seat[k])
        if sc > top_s:
            top = seat[k]
            top_s = sc
    _glob_player(s, me, glob, 0, spots[me], need[me], tl[me])
    _glob_player(s, top, glob, 1, spots[top], need[top], tl[top])
    for w in range(2):
        glob[GI_BEST_GAIN[w]] = best[w, 0]
        glob[GI_BEST_USEFUL[w]] = best[w, 1]
    glob[GI_TOMOVE] = 1 if (s[S_CUR] == me and s[S_OVER] == 0) else 0
    for i in range(5):
        o = s[S_TOWNER + i]
        if o == me:
            glob[GI_TROPHY[i, 0]] = 1
        elif o >= 0:
            glob[GI_TROPHY[i, 1]] = 1
    glob[GI_BAG] = s[S_BAGN]
    for k in range(5):
        sid = s[S_OCEAN + k]
        if sid >= 0:
            for r in range(4):
                glob[GI_OCEAN_RES[k, r]] += LS_NEED[sid, r]
            glob[GI_OCEAN_BONUS[k, LS_CAT[sid]]] = LS_BONUS[sid]
    glob[GI_PLAYERS[npl - 2]] = 1
    for k in range(1, npl):
        p = seat[k]
        glob[GI_SEAT[k - 1, 0]] = 1
        glob[GI_SEAT[k - 1, 1]] = total_score(s, p)
        glob[GI_SEAT[k - 1, 2]] = s[S_VLEFT + p]
        glob[GI_SEAT[k - 1, 3]] = tl[p]
        glob[GI_SEAT[k - 1, 4]] = 1 if p == top else 0


