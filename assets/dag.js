// Здесь описываем функции и методы для ag grid


// Стороим конструкцию

var dagfuncs = (window.dashAgGridFunctions = window.dashAgGridFunctions || {});

dagfuncs.Intl = Intl;

// Русские даты

dagfuncs.RussianDate = function(date) {
    

    if (!date) return "";

    date = new Date(date);
    if (isNaN(date.getTime())) return "";

    return new Intl.DateTimeFormat("ru-RU", {
        day: "2-digit",
        month: "long",
        year: "numeric"
    }).format(date);
};

// Рубли

dagfuncs.RUB = function (number) {
    return Intl.NumberFormat('ru-RU', {
        style: 'currency',
        currency: 'RUB',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(number);
}


// C ед измерения
dagfuncs.FormatWithUnit = function(value, unit='') {
    if (value == null) return "";

    // Форматируем число в русский стиль с 2 знаками после запятой
    let formatted = new Intl.NumberFormat('ru-RU', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);

    return formatted + (unit ? " " + unit : "");
};