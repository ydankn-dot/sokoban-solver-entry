// Push-optimal Sokoban solver, room by room.
//
// A level from this generator is a hallway with one to four rooms off it
// (three or four boxes each). A box pushed out of a room can never be
// pushed back, so the rooms are independent and the level's optimum is the
// sum of its rooms' optima. Each room is solved with A* over pushes (one
// push moves one box one square); a state is the boxes plus the player's
// reachable region, and two safe prunings apply: a box on a dead square
// (one from which no push, ignoring the other boxes, reaches any goal) and
// a frozen 2x2 block (walls and boxes with a box off its goal). The
// heuristic is the cheapest assignment of boxes to goals by Manhattan
// distance, which one push can lower by at most one, so it never overstates
// the pushes left and the first solved state popped is push-optimal.
// No clock, no randomness: the same input always gives the same moves.

const DIRS = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

// Work budget for the whole run, counted in states generated (never the
// clock), so two executions always write the same bytes.
let spent = 0;
const BUDGET = 3_000_000;
const ROOM_STATES = 250_000;

function parse(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

// Squares from which a box alone can be pushed onto some goal: pull back
// from the goals; a square y is live when pushing a box from y lands on a
// live square and the player can stand behind it.
function liveSquares(L) {
  const { w, wall, goals } = L;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

// The player's reachable region from `from` with `occupied` in the way;
// marks it in `seen` with `stamp` and returns its lowest square.
function reach(L, occupied, from, seen, stamp, stack) {
  const { w, wall } = L;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    if (!wall[p - w] && !occupied[p - w] && seen[p - w] !== stamp) { seen[p - w] = stamp; stack[n++] = p - w; }
    if (!wall[p + w] && !occupied[p + w] && seen[p + w] !== stamp) { seen[p + w] = stamp; stack[n++] = p + w; }
    if (!wall[p - 1] && !occupied[p - 1] && seen[p - 1] !== stamp) { seen[p - 1] = stamp; stack[n++] = p - 1; }
    if (!wall[p + 1] && !occupied[p + 1] && seen[p + 1] !== stamp) { seen[p + 1] = stamp; stack[n++] = p + 1; }
  }
  return low;
}

// Whether the box just pushed to `t` sits in a frozen 2x2 block of walls
// and boxes with a box off its goal.
function frozenBlock(L, occupied, goal, t) {
  const { w, wall } = L;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

// Cheapest assignment of boxes to goals by Manhattan distance (brute force
// with branch and bound while there are few boxes, else each box's nearest
// goal summed). One push moves one box one square, so this never overstates
// the pushes left.
function makeHeuristic(L) {
  const { w, goals } = L;
  const n = goals.length;
  const gx = goals.map((g) => g % w);
  const gy = goals.map((g) => Math.floor(g / w));
  if (n <= 6) {
    return (boxes) => {
      let best = Infinity;
      const used = new Uint8Array(n);
      const assign = (i, cost) => {
        if (cost >= best) return;
        if (i === n) { best = cost; return; }
        const b = boxes[i];
        const bx = b % w;
        const by = Math.floor(b / w);
        for (let j = 0; j < n; j++) {
          if (used[j]) continue;
          used[j] = 1;
          assign(i + 1, cost + Math.abs(bx - gx[j]) + Math.abs(by - gy[j]));
          used[j] = 0;
        }
      };
      assign(0, 0);
      return best;
    };
  }
  return (boxes) => {
    let s = 0;
    for (const b of boxes) {
      const bx = b % w;
      const by = Math.floor(b / w);
      let m = Infinity;
      for (let j = 0; j < n; j++) {
        const d = Math.abs(bx - gx[j]) + Math.abs(by - gy[j]);
        if (d < m) m = d;
      }
      s += m;
    }
    return s;
  };
}

// The shortest walk from `from` to `to` around walls and boxes, in lower
// LURD letters (u first on ties); null when `to` is out of reach.
function walkTo(L, occupied, from, to) {
  const { w, wall } = L;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

// The square the player stands on after `moves` from `from` on a board `w` wide.
function endOf(w, from, moves) {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

// The push-optimal solution of `level` in at most `maxStates` states:
// {pushes, moves} (lower letter a step, upper a push), or null when the
// search gives up or the level has no solution.
function pushSearch(L, maxStates) {
  const { w, wall, goals } = L;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(L);
  if (L.boxes.some((b) => !live[b])) return null;
  const hOf = makeHeuristic(L);
  const solved = (boxes) => boxes.every((b) => goal[b]);
  if (solved(L.boxes)) return { pushes: 0, moves: '' };

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const keyOf = (boxes, low) => String.fromCharCode(low, ...boxes);

  // States in creation order: boxes, the player's square after the push
  // that made them, pushes so far, parent, the push (from*size+to), key.
  const sBoxes = [L.boxes.slice()];
  const sPlayer = [L.player];
  const sG = [0];
  const parent = [-1];
  const pushEnc = [-1];
  const sKey = [];
  const closed = new Map();
  const done = new Set();
  for (const b of L.boxes) occupied[b] = 1;
  const low0 = reach(L, occupied, L.player, seen, ++stamp, stack);
  for (const b of L.boxes) occupied[b] = 0;
  sKey[0] = keyOf(L.boxes, low0);
  closed.set(sKey[0], 0);
  spent++;

  // Binary heap on [f, h, sequence, state, g]; f, then h, then age breaks ties.
  const heap = [[hOf(L.boxes), hOf(L.boxes), 0, 0, 0]];
  let seq = 1;
  const less = (a, b) => a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e) => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      const t = heap[i]; heap[i] = heap[p]; heap[p] = t;
      i = p;
    }
  };
  const heapPop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        const t = heap[i]; heap[i] = heap[c]; heap[c] = t;
        i = c;
      }
    }
    return top;
  };

  let found = -1;
  while (heap.length > 0 && found < 0) {
    const [, , , s, g] = heapPop();
    if (g !== sG[s] || done.has(sKey[s])) continue;
    done.add(sKey[s]);
    const boxes = sBoxes[s];
    if (solved(boxes)) { found = s; break; }
    for (const b of boxes) occupied[b] = 1;
    const here = ++stamp;
    reach(L, occupied, sPlayer[s], seen, here, stack);
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(L, occupied, goal, t);
        let key = null;
        let next = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          key = keyOf(next, reach(L, occupied, b, mark, ++stamp, stack));
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || key === null) continue;
        const prevG = closed.get(key);
        if (prevG !== undefined && prevG <= g + 1) continue;
        if (spent >= BUDGET || sBoxes.length >= maxStates) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        spent++;
        closed.set(key, g + 1);
        if (prevG !== undefined) done.delete(key);
        const h = hOf(next);
        sBoxes.push(next);
        sPlayer.push(b);
        sG.push(g + 1);
        parent.push(s);
        pushEnc.push(b * size + t);
        sKey.push(key);
        heapPush([g + 1 + h, h, seq++, sBoxes.length - 1, g + 1]);
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;

  // Rebuild the moves: walk to each push's square, then push.
  const path = [];
  for (let s = found; s > 0; s = parent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = L.player;
  let boxes = L.boxes;
  for (const s of path) {
    const nextBoxes = sBoxes[s];
    const from = boxes.find((b) => !nextBoxes.includes(b));
    const to = nextBoxes.find((b) => !boxes.includes(b));
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d);
    for (const b of boxes) occupied[b] = 1;
    const walk = walkTo(L, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    if (walk === null) return null;
    moves += walk + dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

// The level's rooms: cut the dead squares of the player's row (the hallway:
// a box there reaches no goal) out of the floor, and each part left that
// holds a box is a room. Null when some part holds more or fewer boxes than
// goals, or a box starts in the hallway: then the level is no set of rooms.
function roomsOf(L) {
  const { w, wall, goals, boxes, player } = L;
  const live = liveSquares(L);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i] + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (goals.some((g) => part[g] < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// The level solved room by room: each room's push-optimal solution in turn,
// the other rooms' boxes standing as walls (on their goals once their room
// is done), the player walking on from where the last room left it.
function solveByRooms(L) {
  const rooms = roomsOf(L);
  if (rooms === null) return pushSearch(L, ROOM_STATES);
  let player = L.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = L.wall.slice();
    for (const other of rooms)
      if (other !== room) for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    const sub = {
      w: L.w,
      h: L.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const r = pushSearch(sub, ROOM_STATES);
    if (r === null) return null;
    pushes += r.pushes;
    moves += r.moves;
    player = endOf(L.w, player, r.moves);
    room.done = true;
  }
  return { pushes, moves };
}

// The moves for one level in XSB: '' when it cannot be solved within budget
// (an empty answer is valid, worth no points, and never breaks the format).
export function solve(text) {
  if (spent >= BUDGET) return '';
  const r = solveByRooms(parse(text));
  return r ? r.moves : '';
}
