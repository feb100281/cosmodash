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

// Рубли - ИСПРАВЛЕННАЯ ФУНКЦИЯ
dagfuncs.RUB = function(number) {
    try {
        // Преобразуем в число, если это строка
        const num = typeof number === 'string' ? parseFloat(number) : number;
        
        // Проверяем, что это валидное число
        if (num === null || num === undefined || isNaN(num)) {
            return "0 ₽";
        }
        
        return new Intl.NumberFormat('ru-RU', {
            style: 'currency',
            currency: 'RUB',
            minimumFractionDigits: 0,
            maximumFractionDigits: 0
        }).format(num);
    } catch (error) {
        console.error('RUB formatting error:', error);
        return "0 ₽";
    }
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


// процентики
dagfuncs.FormatPercent = function(value) {
    if (value == null) return "";

    let formatted = new Intl.NumberFormat('ru-RU', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value * 100);

    return formatted + " %";
};

dagfuncs.TwoDecimal = function(value) {
    if (value == null) return "";

    let formatted = new Intl.NumberFormat('ru-RU', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);

    return formatted;
};