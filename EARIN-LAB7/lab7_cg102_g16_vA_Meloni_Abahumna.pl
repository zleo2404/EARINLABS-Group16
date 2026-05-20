% Variant 1 (Lab 7): add / subtract N days from a "DDMM" date in 2024.
% Year fixed at 2024 (leap). Results outside 2024 fail.

% month_days(+Month, -Days) -- 2024 month lengths (leap year).
month_days(1, 31).  month_days(2, 29).  month_days(3, 31).
month_days(4, 30).  month_days(5, 31).  month_days(6, 30).
month_days(7, 31).  month_days(8, 31).  month_days(9, 30).
month_days(10, 31). month_days(11, 30). month_days(12, 31).

% parse_date(+Str, -Day, -Month) -- "DDMM" -> integers.
parse_date(Str, D, M) :-
    string_chars(Str, [D1, D2, M1, M2]),
    number_chars(D, [D1, D2]),
    number_chars(M, [M1, M2]).

% valid_date(+Day, +Month) -- Day fits the given Month in 2024.
valid_date(D, M) :-
    integer(M), M >= 1, M =< 12,
    integer(D), D >= 1,
    month_days(M, MD), D =< MD.

% valid_offset(+N) -- accepted assignment range for added/subtracted days.
valid_offset(N) :-
    integer(N), N >= 0, N =< 366.

% date_to_ordinal(+Day, +Month, -Ord) -- (Day, Month) -> 1..366.
date_to_ordinal(D, 1, D).
date_to_ordinal(D, M, Ord) :-
    M > 1,
    PrevM is M - 1,
    month_days(PrevM, PD),
    date_to_ordinal(D, PrevM, PrevOrd),
    Ord is PrevOrd + PD.

% ordinal_to_date(+Ord, -Day, -Month) -- inverse, fails if Ord is not in 1..366.
ordinal_to_date(Ord, D, M) :-
    integer(Ord),
    ordinal_to_date_(Ord, 1, D, M).

ordinal_to_date_(Ord, M, Ord, M) :-
    M =< 12,
    month_days(M, MD),
    Ord >= 1,
    Ord =< MD,
    !.
ordinal_to_date_(Ord, M, D, OutM) :-
    M < 12,
    month_days(M, MD),
    Ord > MD,
    NextOrd is Ord - MD,
    NextM is M + 1,
    ordinal_to_date_(NextOrd, NextM, D, OutM).

% pad2(+N, -S) -- left-pad with a leading zero.
pad2(N, S) :-
    N < 10,
    !,
    format(string(S), "0~w", [N]).
pad2(N, S) :-
    format(string(S), "~w", [N]).

% make_date_string(+Day, +Month, -Str) -- (Day, Month) -> "DDMM".
make_date_string(D, M, Str) :-
    pad2(D, DS),
    pad2(M, MS),
    string_concat(DS, MS, Str).

% add_date(+DateStr, +N, -Result) -- Computes the new date by adding N days.
add_date(DateStr, N, Result) :-
    valid_offset(N),
    parse_date(DateStr, D, M),
    valid_date(D, M),
    date_to_ordinal(D, M, Ord),
    NewOrd is Ord + N,
    NewOrd >= 1,
    NewOrd =< 366,
    ordinal_to_date(NewOrd, D2, M2),
    make_date_string(D2, M2, Result).

% sub_date(+DateStr, +N, -Result) -- Computes the new date by subtracting N days.
sub_date(DateStr, N, Result) :-
    valid_offset(N),
    parse_date(DateStr, D, M),
    valid_date(D, M),
    date_to_ordinal(D, M, Ord),
    NewOrd is Ord - N,
    NewOrd >= 1,
    NewOrd =< 366,
    ordinal_to_date(NewOrd, D2, M2),
    make_date_string(D2, M2, Result).

% add_date/2, sub_date/2 -- Terminal interface: calculates and prints the result.
add_date(DateStr, N) :-
    add_date(DateStr, N, R),
    format("~q~n", [R]).

sub_date(DateStr, N) :-
    sub_date(DateStr, N, R),
    format("~q~n", [R]).
